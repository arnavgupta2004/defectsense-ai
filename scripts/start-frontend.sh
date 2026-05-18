#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/Frontend"

[ -f .env ] || cp .env.example .env
[ -d node_modules ] || npm install

echo "Starting UI on http://127.0.0.1:8080 (API proxied to :8000)"
exec npm run dev
