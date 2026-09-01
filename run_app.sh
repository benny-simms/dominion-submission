#!/usr/bin/env bash

set -e

echo -e "$Starting Dominion Dynamics Vision Console...$"

echo -e "$Launching FastAPI backend$"
cd report_viewer
cd backend
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

echo -e "$Launching React/Vite frontend$"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo -e "$Initializing frontend$"
sleep 2

echo -e "$Opening browser at: http://localhost:5173$"

if command -v open >/dev/null 2>&1; then
    open http://localhost:5173
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open http://localhost:5173
elif command -v start >/dev/null 2>&1; then
    start http://localhost:5173
fi

wait