# Render 512 MB deployment

Render’s **free** web service has **512 MB RAM**. The API is tuned to fit that limit.

## What we changed

| Setting | Normal | Render / `LOW_MEMORY` |
|---------|--------|------------------------|
| Image size | 224 | **128** |
| Feature layers | layer2 + layer3 | **layer3 only** |
| Coreset ratio | 0.10 | **0.04** |
| Inference chunks | 8192 | **256** |
| Train batch | 16 | **2** |
| Bootstrap images | 12 | **6** |
| Model training | On first boot | **During Docker build** |

`RENDER=true` (set automatically on Render) activates the profile.

## Deploy

1. Use `render.yaml` in the repo, or connect Docker with root `defect-detection/`.
2. Set `CORS_ORIGINS` to your Vercel URL.
3. Attach a **1 GB disk** at `/app/data` (in `render.yaml`) so DB/uploads persist.

## Limits on 512 MB

- **Works:** health check, dashboard, inspect, detect on small images
- **Slow:** first request after sleep (cold start)
- **Risky:** `POST /api/train` with many images (RAM spike) — train locally, upload `patchcore_memory_bank.pt` to disk if needed
- **If OOM:** upgrade to Starter (2 GB) or lower `IMAGE_SIZE=96`

## Verify after deploy

```bash
curl https://YOUR-SERVICE.onrender.com/api/health
curl https://YOUR-SERVICE.onrender.com/api/model/status
```

Expect `"status":"READY"` and `memory_bank_size` > 0.

## Vercel

Set:

```text
VITE_API_BASE_URL=https://YOUR-SERVICE.onrender.com
```

No trailing slash.
