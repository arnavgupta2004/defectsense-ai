# DefectSense Frontend

React + Vite dashboard for the DefectSense API.

## Development

```bash
cp .env.example .env
npm install
npm run dev
```

Open http://localhost:8080. API requests are proxied to `http://localhost:8000` (start the backend separately).

## Production build

```bash
npm run build
```

Or use Docker from the repo root: `docker compose up --build`.

## Vercel

Deploy only this folder to Vercel. Set **Root Directory** to `Frontend` (when the Git repo root is the monorepo root).

**Required env var:** `VITE_API_BASE_URL` = your hosted FastAPI URL (see [../docs/VERCEL.md](../docs/VERCEL.md)).
