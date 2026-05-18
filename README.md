# DefectSense AI

Industrial surface defect detection using **PatchCore** (WideResNet-50-2 features + memory bank) with a **FastAPI** backend and **React** dashboard.

## Quick start (Docker — recommended)

From the **repository root**:

```bash
cp defect-detection/.env.example defect-detection/.env
docker compose up --build
```

| Service  | URL |
|----------|-----|
| Web UI   | http://localhost:8080 |
| API docs | http://localhost:8000/docs |

On first start the API bootstraps sample training images and trains a small memory bank automatically.

## Local development

**Requirements:** Python **3.11+** (macOS default 3.9 will not work), Node 18+.

### Option A — helper scripts (easiest)

Terminal 1:

```bash
./scripts/start-backend.sh
```

Terminal 2:

```bash
./scripts/start-frontend.sh
```

Open http://127.0.0.1:8080

### Option B — manual

**Backend** (use `python3.12` or `python3.11`, not system 3.9):

```bash
cd defect-detection
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/bootstrap_data.py
python -m training.train_patchcore
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Frontend:**

```bash
cd Frontend
cp .env.example .env
npm install
npm run dev
```

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `ImportError: TRAINING_STATE` / circular import | Pull latest code (`app/state.py` fix) |
| `numpy` / `torch` install fails | Use Python 3.11+ (`brew install python@3.12`) |
| Dashboard shows API error | Start backend on port 8000 first |
| `docker compose` fails | Start **Docker Desktop**, then retry |
| Inspection returns 503 | Run training once (`python -m training.train_patchcore`) |

## Project layout

```text
repository root/
├── Dockerfile.api      # Render: Docker build from repo root
├── defect-detection/   # FastAPI + PatchCore pipeline
├── Frontend/           # React + Vite UI
├── scripts/
├── docs/
├── docker-compose.yml
└── render.yaml
```

## API highlights

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Upload image |
| POST | `/api/detect/{image_id}` | Run inference |
| POST | `/api/train` | Train memory bank (background) |
| GET | `/api/dashboard` | Dashboard aggregates |
| GET | `/api/results` | List inspections |
| PATCH | `/api/model/threshold` | Update detection threshold |

## Training on your data

Place **normal-only** images in:

```text
defect-detection/data/custom/train/good/
```

Then:

```bash
python -m training.train_patchcore
# or POST /api/train
```

## Deploy on Vercel (frontend)

The UI deploys to **Vercel**; the API must run elsewhere (Render, Railway, Fly, Docker VPS).

See **[docs/VERCEL.md](docs/VERCEL.md)** for step-by-step instructions.

Quick summary:

1. Deploy API with `render.yaml` (uses root `Dockerfile.api`) or Docker per [docs/VERCEL.md](docs/VERCEL.md).
2. Vercel → Root Directory: `Frontend`
3. Set `VITE_API_BASE_URL=https://your-api-host`
4. Set API `CORS_ORIGINS` to your `*.vercel.app` URL.

## Production notes (self-hosted / Docker)

- Set `CORS_ORIGINS` to your frontend domain in `defect-detection/.env`
- Persist Docker volumes: `model-artifacts`, `db-data`, `upload-data`
- For GPU: set `MODEL_DEVICE=cuda` and use a CUDA PyTorch base image

## Tests

```bash
cd defect-detection && pytest
cd ../Frontend && npm test
```
