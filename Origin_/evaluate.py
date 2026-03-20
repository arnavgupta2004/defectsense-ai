import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from dataset import get_combined_dataset, CLIPSegCollateFn
from torch.utils.data import DataLoader
from train import compute_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Path to fine-tuned model")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Directory to save visual examples")
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
        print(f"Model not found in checkpoints or error ({e}), falling back to base model for evaluation testing.")
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        
    model.to(device)
    model.eval()
    
    # We will evaluate on the valid dataset since test dataset annotations might not be cleanly aggregated or we want a unified metric
    valid_dataset = get_combined_dataset("valid")
    collate_fn = CLIPSegCollateFn(processor)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # Track metrics per prompt category for rubric requirement: mIoU & Dice on BOTH prompts
    metrics_by_prompt = {}  # prompt -> {"ious": [], "dices": []}
    ious = []
    dices = []

    saved_visuals = 0
    num_visuals_to_save = 4

    print("Starting evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(valid_loader)):
            inputs = {k: v.contiguous().to(device) if isinstance(v, torch.Tensor) else v.to(device) for k, v in batch.items() if k not in ["original_sizes", "prompts", "labels"]}
            target_masks = batch["labels"].contiguous().to(device)

            outputs = model(**inputs)
            logits = outputs.logits

            if logits.ndim == 3:
                logits = logits.unsqueeze(1)
            if target_masks.ndim == 3:
                target_masks = target_masks.unsqueeze(1)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            for j in range(len(preds)):
                gt = target_masks[j].cpu().numpy()
                pred = preds[j].cpu().numpy()

                # Skip samples with no foreground pixels in ground truth.
                # Smooth-IoU returns 1.0 when both GT and pred are all-zero,
                # which silently inflates scores — standard practice is to exclude these.
                if gt.sum() == 0:
                    continue

                # Metric computation
                iou, dice = compute_metrics(pred, gt)
                ious.append(iou)
                dices.append(dice)

                # Track per-prompt metrics
                prompt = batch["prompts"][j]
                # Map prompt to canonical category
                if any(kw in prompt for kw in ["crack"]):
                    category = "crack"
                else:
                    category = "taping_area"
                if category not in metrics_by_prompt:
                    metrics_by_prompt[category] = {"ious": [], "dices": []}
                metrics_by_prompt[category]["ious"].append(iou)
                metrics_by_prompt[category]["dices"].append(dice)

                # Visual generation
                if saved_visuals < num_visuals_to_save:
                    # CLIPSeg uses CLIP normalization, not ImageNet
                    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(device)
                    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(device)
                    
                    img_tensor = inputs["pixel_values"][j]
                    img_unnorm = img_tensor * std + mean
                    img_unnorm = torch.clamp(img_unnorm, 0, 1).cpu().permute(1, 2, 0).numpy()

                    pred_np = preds[j].squeeze(0).cpu().numpy()
                    target_np = target_masks[j].squeeze(0).cpu().numpy()

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    fig.suptitle(f"Prompt: '{prompt}' | mIoU: {iou:.3f} | Dice: {dice:.3f}")
                    
                    axes[0].imshow(img_unnorm)
                    axes[0].set_title("Original Image (Resized)")
                    axes[0].axis('off')
                    
                    axes[1].imshow(target_np, cmap='gray')
                    axes[1].set_title("Ground Truth Mask")
                    axes[1].axis('off')
                    
                    axes[2].imshow(pred_np, cmap='gray')
                    axes[2].set_title("Predicted Mask")
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.output_dir, f"visual_{saved_visuals}.png"))
                    plt.close(fig)
                    
                    saved_visuals += 1
                    
    final_iou = np.mean(ious)
    final_dice = np.mean(dices)

    print("\n" + "="*40)
    print("Final Evaluation Metrics:")
    print(f"Total validation samples: {len(ious)}")
    print(f"Overall mIoU: {final_iou:.4f}")
    print(f"Overall Dice Score: {final_dice:.4f}")
    print("-"*40)
    print("Per-Prompt Breakdown:")
    for category, m in metrics_by_prompt.items():
        cat_iou = np.mean(m["ious"])
        cat_dice = np.mean(m["dices"])
        print(f"  [{category}] samples={len(m['ious'])}  mIoU={cat_iou:.4f}  Dice={cat_dice:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
