import torch
from torch.utils.data import DataLoader
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch.nn as nn
from dataset import get_combined_dataset, CLIPSegCollateFn
import numpy as np
from tqdm import tqdm
import os
import argparse

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_metrics(pred_masks, true_masks):
    # Flatten masks
    pred_masks = pred_masks.flatten()
    true_masks = true_masks.flatten()
    
    intersection = (pred_masks * true_masks).sum()
    union = pred_masks.sum() + true_masks.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    dice = (2. * intersection + 1e-6) / (pred_masks.sum() + true_masks.sum() + 1e-6)
    
    return iou.item(), dice.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.to(device)
    
    train_dataset = get_combined_dataset("train")
    valid_dataset = get_combined_dataset("valid")
    
    collate_fn = CLIPSegCollateFn(processor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_iou = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            inputs = {k: v.contiguous().to(device) if isinstance(v, torch.Tensor) else v.to(device) for k, v in batch.items() if k not in ["original_sizes", "prompts", "labels"]}
            target_masks = batch["labels"].contiguous().to(device)
            
            outputs = model(**inputs)
            # The model outputs raw logits
            logits = outputs.logits
            if logits.ndim == 3:
                logits = logits.unsqueeze(1) # B, 1, H, W
            # Check target dimension
            if target_masks.ndim == 3:
                target_masks = target_masks.unsqueeze(1) # B, 1, H, W
            
            # Make sure sizes match. Model output might be exactly 352x352
            loss = criterion(logits, target_masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        print(f"Average Train Loss: {train_loss / len(train_loader):.4f}")
        
        # Validation
        model.eval()
        valid_loss = 0.0
        ious = []
        dices = []
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Valid]"):
                inputs = {k: v.contiguous().to(device) if isinstance(v, torch.Tensor) else v.to(device) for k, v in batch.items() if k not in ["original_sizes", "prompts", "labels"]}
                target_masks = batch["labels"].contiguous().to(device)
                
                outputs = model(**inputs)
                logits = outputs.logits
                if logits.ndim == 3:
                    logits = logits.unsqueeze(1)
                if target_masks.ndim == 3:
                    target_masks = target_masks.unsqueeze(1)
                    
                loss = criterion(logits, target_masks)
                valid_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                for i in range(len(preds)):
                    gt = target_masks[i].cpu().numpy()
                    # Skip empty GT masks — smooth-IoU returns 1.0 for both-zero case,
                    # which would corrupt best-model selection.
                    if gt.sum() == 0:
                        continue
                    iou, dice = compute_metrics(preds[i].cpu().numpy(), gt)
                    ious.append(iou)
                    dices.append(dice)
                    
        val_iou = np.mean(ious)
        val_dice = np.mean(dices)
        print(f"Valid Loss: {valid_loss / len(valid_loader):.4f}")
        print(f"Valid mIoU: {val_iou:.4f} | Valid Dice: {val_dice:.4f}")
        
        scheduler.step()

        if val_iou > best_iou:
            best_iou = val_iou
            print("Saving best model...")
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
