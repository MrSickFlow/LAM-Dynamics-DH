#!/bin/bash
PROJECT_DIR="/Users/Omistaja/defence3/LAM-Dynamics-DH"
LOG="$PROJECT_DIR/.claude/backend.log"

pkill -f "uvicorn ipb_backend.main:app" 2>/dev/null || true
sleep 0.5

cd "$PROJECT_DIR"
nohup env PYTHONPATH=src .venv/bin/uvicorn ipb_backend.main:app \
  --host 0.0.0.0 --port 8000 \
  >> "$LOG" 2>&1 &

echo "Backend restarted (PID $!), log: $LOG"
