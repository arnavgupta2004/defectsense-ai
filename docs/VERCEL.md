# Deploy on Vercel

Vercel hosts the **React frontend only**. The PatchCore / PyTorch API cannot run on Vercel serverless (size limits, no GPU, no long-running model in memory).

Use this split:

| Part | Platform | Why |
|------|----------|-----|
| **Frontend** | Vercel | Static Vite build |
| **API** | Render, Railway, Fly.io, or a VPS | FastAPI + PyTorch + SQLite/uploads |

---

## 1. Deploy the API (Render — quickest)

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New** → **Blueprint** (or Web Service from Docker).
3. Point to `render.yaml` at the **repository root**, or connect Docker manually:
   - **Blueprint:** leave the service **Root Directory** in the dashboard **empty** (the YAML uses repo-root `dockerfilePath` / `dockerContext`). If you set Root Directory to `defect-detection` **and** keep paths like `./defect-detection/Dockerfile`, Render resolves `defect-detection/defect-detection` and the deploy fails.
   - **Manual Docker (no blueprint):** either leave **Root Directory** empty and set **Dockerfile path** to `defect-detection/Dockerfile`, **or** set **Root Directory** to `defect-detection` and **Dockerfile path** to `./Dockerfile` — pick one style, not both.
4. Set environment variable:
   ```text
   CORS_ORIGINS=https://your-app.vercel.app,http://localhost:8080
   ```
5. After deploy, copy the service URL, e.g. `https://defectsense-api.onrender.com`.

Free tier: **512 MB RAM**. This repo’s Docker image uses a **low-memory profile** (128px images, smaller memory bank, model baked at build time). Expect:

- Cold start ~30–60s after sleep
- Inference ~5–15s per image on CPU
- Do **not** run `/api/train` on the free tier unless you have few images (training spikes RAM)

If the service crashes with **OOM**, upgrade to Render **Starter (2 GB)** or set `IMAGE_SIZE=96` and `CORESET_SAMPLING_RATIO=0.02`.

---

## 2. Deploy the frontend (Vercel)

### Via dashboard

1. [vercel.com/new](https://vercel.com/new) → Import your Git repo.
2. **Root Directory:** `Frontend`
3. Framework: **Vite** (auto-detected).
4. **Environment variables** (Production):

   | Name | Value |
   |------|--------|
   | `VITE_API_BASE_URL` | `https://defectsense-api.onrender.com` |

5. Deploy.

### Via CLI

```bash
cd Frontend
npm i -g vercel
vercel
# Set VITE_API_BASE_URL when prompted, or in Project Settings → Environment Variables
vercel --prod
```

---

## 3. Wire CORS on the API

Update the API host’s `CORS_ORIGINS` to include your Vercel URL:

```text
https://defectsense-ai.vercel.app,https://defectsense-ai-xxx.vercel.app
```

Redeploy or restart the API after changing env vars.

---

## 4. Verify

1. Open `https://your-app.vercel.app`
2. Dashboard should load (not “Could not load dashboard”).
3. **Model** page should show **Model Trained** (after API finished first-boot training).
4. **Inspect** → upload an image → run inspection.

API health check: `https://your-api-host/api/health`

---

## Vercel project settings (reference)

| Setting | Value |
|---------|--------|
| Root Directory | `Frontend` |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Install Command | `npm ci` |

`vercel.json` in the Frontend folder configures SPA routing for React Router.

---

## What not to do

- Do not deploy `defect-detection/` as Vercel serverless Python — PyTorch exceeds limits and cold starts will fail.
- Do not leave `VITE_API_BASE_URL` empty in production — the UI will call the Vercel domain and get 404 on `/api/*`.
