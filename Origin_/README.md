# Drywall QA Segmenter

This repository contains the fine-tuning setup, inference, and evaluation scripts for a text-conditioned segmentation model designed to identify recording/taping areas and cracks on drywall.

## Project Structure
- `dataset.py`: Parses COCO JSON annotations and implements a PyTorch Dataset for Hugging Face `CLIPSeg`. 
- `train.py`: Fine-tunes the base `CIDAS/clipseg-rd64-refined` model using `BCEWithLogitsLoss`.
- `evaluate.py`: Calculates mIoU and Dice Score metrics, and generates visual comparison plots.
- `inference.py`: Processes a folder of images using the given prompts and outputs single-channel PNG masks.

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Datasets**
   Ensure `Dataset/Drywall-Join-Detect` and `Dataset/cracks` are in the project root. The annotation format must be COCO JSON (`_annotations.coco.json`).

3. **Fine-Tuning the Model**
   Run the training script (default uses CPU due to PyTorch MPS convolution view size bugs on M1/M2/M3 chips):
   ```bash
   python train.py --epochs 5 --batch_size 4 --device cpu --output_dir checkpoints
   ```
   *Random Seed used is 42.*

4. **Inference**
   Generate pixel-perfect (`{0, 255}`) PNG masks:
   ```bash
   python inference.py --model_dir checkpoints --output_dir predictions
   ```

5. **Evaluation**
   Validate metrics and generate visualizations:
   ```bash
   python evaluate.py --model_dir checkpoints --output_dir evaluation
   ```
