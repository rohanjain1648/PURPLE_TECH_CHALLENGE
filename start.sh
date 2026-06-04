#!/bin/sh
# Startup script for Railway and local Docker.
# Railway injects $PORT at runtime; falls back to 8000 for local use.
set -e
PORT="${PORT:-8000}"
echo "Starting uvicorn on port $PORT"
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers 1
