## Industrial Defect Detection System (PatchCore)

This project implements an **industrial surface defect detection and localization system** using **PatchCore anomaly detection** with a **WideResNet-50-2** backbone, exposed via a **FastAPI** backend.

### Architecture (High-Level)

```text
           +-----------------+
           |   Client (UI)   |
           +--------+--------+
                    |
                    v
           +-----------------+
           |    FastAPI      |
           |  (app/main.py)  |
           +--------+--------+
                    |
     +--------------+-----------------------+
     |              |                       |
     v              v                       v
+---------+   +-----------+        +-----------------+
| Upload  |   |  Detect   |        |  Training/Eval  |
| /upload |   | /detect   |        |  training/*.py  |
+----+----+   +-----+-----+        +-----------------+
     |              |
     v              v
+----+---------------------------+
|   Core Pipeline (app/core)     |
|  - preprocessor.py             |
|  - feature_extractor.py        |
|  - patchcore.py (memory bank)  |
|  - anomaly_map.py              |
|  - postprocessor.py            |
|  - visualizer.py               |
+--------------------------------+
                |
                v
        +---------------+
        |   SQLite DB   |
        | (SQLAlchemy)  |
        +---------------+
```

### Key Components

- **Preprocessor**: Resize, denoise, normalize, convert to tensors.
- **Feature Extractor**: Pretrained WideResNet-50-2 intermediate features (layer2, layer3).
- **PatchCore Wrapper**: Uses `anomalib` PatchCore with explicit memory bank save/load.
- **Anomaly Map**: Upsamples patch-level scores, smooths, normalizes.
- **Postprocessor**: Thresholding, classification (NORMAL/DEFECTIVE), contour-based localization.
- **Visualizer**: Jet heatmap overlay, bounding boxes, score/status text; outputs base64.
- **FastAPI**: REST API for upload, detect, train, results, health/model status.
- **Training/Eval**: Standalone scripts to build memory bank and compute metrics.

---

## Setup

### 1. Create and activate environment

```bash
cd defect-detection
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU, install a CUDA-enabled PyTorch build first, then `anomalib` with the appropriate extra (e.g. `anomalib[cu124]`).

### 3. Configure environment

Copy the example env file:

```bash
cp .env.example .env
```

Edit `.env` as needed (model paths, threshold, DB URL, upload dir, etc.).

---

## Running the API

From the `defect-detection` directory:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open the interactive docs at `http://localhost:8000/docs`.

---

## API Endpoints

- **POST `/api/upload`**
  - Upload an image file.
  - **Response**:
    - `image_id`: UUID of stored image.
    - `filename`: Stored filename.

- **POST `/api/detect/{image_id}`**
  - Run defect detection on a previously uploaded image.
  - **Response (example)**:

```json
{
  "image_id": "uuid",
  "filename": "product_001.jpg",
  "status": "DEFECTIVE",
  "anomaly_score": 0.847,
  "threshold": 0.5,
  "defect_regions": [
    {
      "bbox": [10, 32, 48, 20],
      "severity": "HIGH",
      "area_percent": 3.2
    }
  ],
  "annotated_image": "base64string",
  "inference_time_ms": 43,
  "timestamp": "2024-01-15T10:30:00"
}
```

- **POST `/api/train`**
  - Triggers background PatchCore training using `data/custom/train/good/` (or optional custom path).
  - **Request body**:

```json
{ "dataset_path": "optional/custom/train/good" }
```

- **GET `/api/results/{image_id}`**
  - Fetch a stored detection result.

- **GET `/api/results`**
  - List results with query params:
    - `status` (`NORMAL` / `DEFECTIVE`)
    - `filename_contains`
    - `limit`

- **GET `/api/model/status`**
  - Returns model training status and latest metrics.

- **GET `/api/health`**
  - Health check.

---

## Example `curl` Calls

### Upload an image

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@data/samples/product_001.jpg"
```

### Run detection

```bash
curl -X POST "http://localhost:8000/api/detect/{image_id}"
```

### Trigger training

```bash
curl -X POST "http://localhost:8000/api/train" \
  -H "Content-Type: application/json" \
  -d '{ "dataset_path": "data/custom/train/good" }'
```

### List results

```bash
curl "http://localhost:8000/api/results?status=DEFECTIVE&limit=20"
```

---

## Training on Custom Data

1. Place **only normal (non-defective)** images in:

```text
data/custom/train/good/
```

2. Optionally prepare test data:

```text
data/custom/test/good/
data/custom/test/defective/
```

3. Run the training script:

```bash
python -m training.train_patchcore
```

4. The memory bank is saved to the path given by `MODEL_MEMORY_BANK_PATH` in `.env` (default: `./artifacts/patchcore_memory_bank.pt`).

5. Start the API and call `/api/detect/{image_id}` as usual – it will load the memory bank automatically.

---

## Evaluating the Model

Use the evaluation script:

```bash
python -m training.evaluate
```

This computes:

- **Image-level AUROC**
- **Pixel-level AUROC**
- **F1 score** at optimal threshold
- **Average Precision**
- **Per-Region Overlap (PRO)** (approximation)
- Confusion matrix and sample counts

The report is written to `evaluation_report.json`.

---

## Metrics Tracked

- **Image-level AUROC**
- **Pixel-level AUROC**
- **F1 score** and **best threshold**
- **Average Precision**
- **Inference latency per image** (included in API responses)
- **Memory bank size** (placeholder stored in model/training status)

---

## MVTec AD Dataset

Use the helper script to download and unpack all MVTec AD categories:

```bash
cd data
python download_mvtec.py
```

You can then point `training/config.yaml` or `/api/train` to the appropriate MVTec category directories (e.g. `mvtec/bottle/train/good`, `mvtec/bottle/test`).

---

## Synthetic Defect Generation

To bootstrap defective samples from normal images:

1. Place normal images into `data/custom/test/good/`.
2. Run:

```bash
python scripts/generate_synthetic_defects.py
```

Synthetic defective images with scratches, blobs, cracks, and discoloration are written to:

```text
data/custom/test/defective/
```

You can then use `training/evaluate.py` to measure performance on this synthetic test set.

---

## Running Tests

From the `defect-detection` directory:

```bash
pytest
```

This runs:

- Preprocessor unit tests
- PatchCore wrapper sanity tests
- Basic API tests (health + upload)

