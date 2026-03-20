import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import torch.nn.functional as F

class DrywallSegmentationDataset(Dataset):
    def __init__(self, data_dir, split, prompts):
        """
        data_dir: string, path to the dataset folder
        split: train, valid, test
        prompts: list of strings to use as text prompts
        """
        self.data_dir = data_dir
        self.split = split
        self.prompts = prompts
        self.image_dir = os.path.join(data_dir, split)
        self.annotation_file = os.path.join(self.image_dir, "_annotations.coco.json")
        
        self.coco = COCO(self.annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        
    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            try:
                pixel_mask = self.coco.annToMask(ann)
                mask = np.maximum(mask, pixel_mask)
            except Exception:
                continue
            
        prompt = random.choice(self.prompts)
        
        return image, prompt, mask

class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        
    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)
        
    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - len(self.dataset1)]

def get_combined_dataset(split):
    prompts1 = ["segment taping area", "segment joint/tape", "segment drywall seam"]
    prompts2 = ["segment crack", "segment wall crack"]
    
    ds1_path = "Dataset/Drywall-Join-Detect"
    ds2_path = "Dataset/cracks"
    
    datasets = []
    if os.path.exists(os.path.join(ds1_path, split, "_annotations.coco.json")):
        ds1 = DrywallSegmentationDataset(ds1_path, split, prompts1)
        datasets.append(ds1)
    if os.path.exists(os.path.join(ds2_path, split, "_annotations.coco.json")):
        ds2 = DrywallSegmentationDataset(ds2_path, split, prompts2)
        datasets.append(ds2)
        
    if len(datasets) == 0:
        return None
    elif len(datasets) == 1:
        return datasets[0]
    else:
        return CombinedDataset(datasets[0], datasets[1])

class CLIPSegCollateFn:
    def __init__(self, processor):
        self.processor = processor
        
    def __call__(self, batch):
        images = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        masks = [item[2] for item in batch]
        
        inputs = self.processor(
            text=texts,
            images=images,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Original size for metric computation
        original_sizes = [m.shape for m in masks]
        
        # Resize masks to the model's expected size (352x352 for CLIPSeg)
        target_size = inputs["pixel_values"].shape[-2:]
        
        # Pad masks that don't match exactly or convert directly
        resized_masks = []
        for mask in masks:
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # 1, 1, H, W
            mask_resized = F.interpolate(mask_tensor, size=target_size, mode="nearest").squeeze(0).squeeze(0)
            resized_masks.append(mask_resized)
            
        inputs["labels"] = torch.stack(resized_masks)
        inputs["original_sizes"] = original_sizes
        inputs["prompts"] = texts
        
        return inputs
