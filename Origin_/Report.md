# Drywall QA Segmentation — Technical Report

---

## 1. Methodology

### Model Selection
For this task, I fine-tuned **CLIPSeg** (`CIDAS/clipseg-rd64-refined`), a text-conditioned image segmentation model available via Hugging Face Transformers.

Unlike standard semantic segmentation models (e.g., U-Net, DeepLabV3) which require fixed output class heads, CLIPSeg accepts an image and a free-form natural language text prompt and produces a continuous segmentation map. This is an ideal fit for the assignment requirement — the same model must respond to distinct prompts ("segment crack", "segment taping area", etc.) without architectural changes per category.

**CLIPSeg Architecture Overview:**
- A frozen CLIP vision encoder extracts image features
- A frozen CLIP text encoder encodes the prompt
- A lightweight, trainable FiLM-conditioned convolutional decoder fuses both modalities and produces a `352×352` segmentation logit map
- The decoder is the primary target of fine-tuning; the CLIP backbone provides strong zero-shot priors

### Training Setup
| Hyperparameter | Value |
|---|---|
| Base model | `CIDAS/clipseg-rd64-refined` |
| Optimizer | AdamW |
| Learning rate | 1e-5 |
| LR schedule | Cosine Annealing (T_max = 10) |
| Loss function | BCEWithLogitsLoss |
| Epochs | 10 |
| Batch size | 4 |
| Random seed | 42 |
| Device | CPU (MPS excluded — known PyTorch view/stride bug in CLIPSeg decoder) |

**Best checkpoint**: Epoch 5, saved based on validation mIoU on non-empty ground truth samples.

**Prompt augmentation**: During training, each image is randomly paired with one prompt from a set of semantically equivalent variants (e.g., "segment crack", "segment wall crack"). This prevents the model from overfitting to a single phrasing and improves robustness at inference time.

---

## 2. Data Preparation

### Datasets
Two datasets were used, both in COCO JSON annotation format from Roboflow:

| Dataset | Task | Train | Valid | Test |
|---|---|---|---|---|
| Drywall-Join-Detect | Taping area segmentation | 154 | 44 | — |
| Cracks | Crack segmentation | 2,269 | 218 | 109 |

### Preprocessing Pipeline
1. **COCO annotation parsing**: Each image's annotation polygons/RLE masks were decoded using `pycocotools` and merged into a single binary mask via pixel-wise maximum (handles overlapping annotations).
2. **Mask binarisation**: Output masks are `{0, 1}` float tensors at training time, upscaled to `{0, 255}` uint8 PNGs at inference.
3. **Image & mask resizing**: CLIPSeg's processor resizes all inputs to `352×352`. Masks are independently resized using nearest-neighbour interpolation to preserve binary sharpness.
4. **Normalisation**: Images are normalised using CLIP's statistics (`mean=[0.48145, 0.45783, 0.40821]`, `std=[0.26863, 0.26130, 0.27577]`), applied internally by the CLIPSeg processor.
5. **Prompt assignment**: Each sample is paired at load time with a randomly selected prompt from the category-specific list:
   - Taping area: `["segment taping area", "segment joint/tape", "segment drywall seam"]`
   - Cracks: `["segment crack", "segment wall crack"]`

### Dataset Observations
- The Drywall-Join-Detect **validation split contained no mask annotations** (all-zero ground truth masks). This was detected during evaluation — smooth-IoU silently returns 1.0 when both prediction and GT are all-zero, which would have inflated reported metrics. Empty-GT samples are excluded from metric computation.
- The Cracks dataset is significantly larger (~15× more training images) and dominates the combined training loss.

---

## 3. Results

### Quantitative Metrics
Metrics computed on the validation set using the best checkpoint (epoch 5, seed=42). Empty ground-truth samples excluded.

| Prompt Category     | Val Samples | mIoU   | Dice Score |
|---------------------|-------------|--------|------------|
| segment crack       | 201         | 0.4285 | 0.5758     |
| segment taping area | 0 *(no valid annotations)* | — | — |
| **Overall** | **201** | **0.4285** | **0.5758** |

### Per-Epoch Training Progression

| Epoch | Train Loss | Val Loss | Val mIoU | Val Dice |
|-------|-----------|----------|----------|----------|
| 1     | 0.0920    | 0.0497   | 0.3976   | 0.5397   |
| 2     | 0.0749    | 0.0449   | 0.4227   | 0.5709   |
| 3     | 0.0714    | 0.0441   | 0.4284   | 0.5772   |
| 4     | 0.0693    | 0.0438   | 0.4183   | 0.5665   |
| **5** ✓ | **0.0679** | **0.0425** | **0.4291** | **0.5761** |
| 6     | 0.0668    | 0.0422   | 0.4250   | 0.5713   |
| 7     | 0.0662    | 0.0424   | 0.4201   | 0.5656   |
| 8     | 0.0659    | 0.0419   | 0.4277   | 0.5731   |
| 9     | 0.0656    | 0.0427   | 0.4226   | 0.5675   |
| 10    | 0.0655    | 0.0416   | 0.4256   | 0.5706   |

The model converges by epoch 5 and plateaus thereafter, indicating the decoder has reached the limit of what can be learned from this data at this learning rate.

### Qualitative Results
Side-by-side visual examples (Original Image | Ground Truth Mask | Predicted Mask) are included below and saved in the `/evaluation` directory.

---

## 4. Failure Cases & Potential Solutions

### Failure Case 1 — Low-Resolution Backbone Struggles with Thin Cracks
**Problem**: CLIPSeg operates at a fixed `352×352` resolution. Wall crack images that were originally captured at 1080p or higher must be aggressively downscaled before entering the Vision Transformer backbone. Very thin hairline cracks (1–3 pixels wide at native resolution) can disappear entirely after downscaling, making them undetectable.

**Potential solutions**:
- Use a **sliding window inference** strategy — run the model on high-resolution tiles and stitch predictions back together.
- Replace CLIPSeg with a **higher-resolution text-conditioned model** such as SEEM or SAM (Segment Anything Model) with a text adapter, both of which support larger input sizes.
- Apply **CLAHE (Contrast Limited Adaptive Histogram Equalisation)** pre-processing to enhance crack visibility before downscaling.

---

### Failure Case 2 — Taping Area Validation Data Had No Annotations
**Problem**: The Drywall-Join-Detect valid split contained zero annotated mask pixels across all 44 images. This means quantitative evaluation of taping area performance was impossible on the provided splits. The model may have learned taping area segmentation during training but we cannot verify it from validation metrics alone.

**Potential solutions**:
- Re-annotate or obtain a corrected version of the Drywall-Join-Detect validation split.
- Evaluate on the **training split** as a proxy (with the caveat of potential overfitting).
- Use **qualitative human evaluation** on held-out images as an interim measure.

---

### Failure Case 3 — Class Imbalance Between Datasets
**Problem**: The Cracks dataset has ~15× more training images than Drywall-Join-Detect (2,269 vs 154). The combined training loss is dominated by crack samples, potentially undertraining the taping area decoder capacity.

**Potential solutions**:
- Apply **weighted sampling** during training to ensure equal representation per dataset per batch.
- Use **separate loss weighting** — upweight the taping area loss by a factor proportional to the imbalance.
- Augment the Drywall-Join-Detect dataset with geometric and photometric transforms (flips, rotations, brightness jitter) to artificially expand its size.

---

### Failure Case 4 — Prompt Ambiguity at Boundaries
**Problem**: "Taping area" and "drywall seam" are visually similar regions. In blurry or low-contrast images, the model sometimes over-segments background wall texture when prompted with "segment joint/tape", treating smooth wall surfaces as tape.

**Potential solutions**:
- Fine-tune with **hard negative examples** — images of plain drywall explicitly labelled as background.
- Use **test-time prompt ensembling** — average predictions across multiple prompt variants and threshold the mean.
- Add a **post-processing morphological step** (erosion followed by dilation) to suppress scattered false-positive noise blobs.

---

## Runtime & Footprint
| Metric | Value |
|---|---|
| Training time | ~3.5 hours, 10 epochs, CPU |
| Inference time | ~800ms per image per prompt, CPU |
| Model size | ~130 MB |
| Random seed | 42 |
