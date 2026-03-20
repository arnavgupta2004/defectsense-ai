import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import argparse
from tqdm import tqdm
import torch.nn.functional as F

def get_prompts_for_dataset(dataset_type):
    if dataset_type == "drywall":
        return ["segment taping area", "segment joint/tape", "segment drywall seam"]
    elif dataset_type == "cracks":
        return ["segment crack", "segment wall crack"]
    else:
        # Combined evaluation or just custom prompts
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Path to fine-tuned model")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save masks")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use. MPS is intentionally excluded — PyTorch MPS has a known view/stride bug inside CLIPSeg's transposed convolution.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    print(f"Loading model from {args.model_dir}...")
    try:
        processor = CLIPSegProcessor.from_pretrained(args.model_dir)
        model = CLIPSegForImageSegmentation.from_pretrained(args.model_dir)
    except Exception as e:
        print(f"Model not found in checkpoints or error ({e}), falling back to base model for testing inference.")
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        
    model.to(device)
    model.eval()
    
    # Let's run inference on the "test" split of cracks, and "valid" split of Drywall (since Drywall no test split)
    datasets_to_run = [
        {"dir": "Dataset/Drywall-Join-Detect/valid", "type": "drywall"},
        {"dir": "Dataset/cracks/test", "type": "cracks"}
    ]
    
    for ds_info in datasets_to_run:
        img_dir = ds_info["dir"]
        ds_type = ds_info["type"]
        
        if not os.path.exists(img_dir):
            print(f"Directory {img_dir} does not exist, skipping.")
            continue
            
        prompts = get_prompts_for_dataset(ds_type)
        if not prompts:
            continue
            
        print(f"Processing {img_dir} with prompts: {prompts}")
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        with torch.no_grad():
            for img_file in tqdm(img_files):
                img_path = os.path.join(img_dir, img_file)
                image = Image.open(img_path).convert("RGB")
                orig_size = (image.height, image.width)
                
                # Image id without extension
                img_id = os.path.splitext(img_file)[0]
                
                # We can run all prompts in a batch for the same image
                inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
                inputs = {k: v.contiguous().to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                outputs = model(**inputs)
                
                logits = outputs.logits
                if len(prompts) == 1:
                    logits = logits.unsqueeze(0)
                    
                # logits shape: (num_prompts, 352, 352)
                logits = logits.unsqueeze(1) # (num_prompts, 1, 352, 352)
                
                # Resize back to original
                logits_resized = F.interpolate(logits, size=orig_size, mode="bilinear", align_corners=False)
                logits_resized = logits_resized.squeeze(1) # (num_prompts, H, W)
                
                probs = torch.sigmoid(logits_resized)
                preds = (probs > 0.5).float()
                
                for i, prompt in enumerate(prompts):
                    pred_mask = preds[i].cpu().numpy()
                    pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
                    
                    # Convert prompt to slug format: spaces to underscores, remove slashes
                    prompt_slug = prompt.replace(" ", "_").replace("/", "_")
                    
                    out_filename = f"{img_id}__{prompt_slug}.png"
                    out_path = os.path.join(args.output_dir, ds_type, out_filename)
                    
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    Image.fromarray(pred_mask_uint8).save(out_path)
                    
    print("Inference completed!")

if __name__ == "__main__":
    main()
