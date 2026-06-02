#!/bin/bash
set -e

echo "🔄 Initializing database..."
python data.py

echo "🚀 Starting FastAPI server..."
exec uvicorn api:app --host 0.0.0.0 --port 8000
