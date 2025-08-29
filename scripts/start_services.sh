#!/bin/bash
# Start services script for Phishing Triage System
# This script starts both the backend and frontend services

# Set the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Function to check if a port is in use
port_in_use() {
  lsof -i:"$1" >/dev/null 2>&1
  return $?
}

# Function to kill a process using a specific port
kill_port_process() {
  echo "Killing process on port $1..."
  lsof -ti:"$1" | xargs kill -9 2>/dev/null || true
}

# Kill any existing processes on our ports
if port_in_use 8001; then
  echo "Port 8001 is already in use."
  kill_port_process 8001
fi

if port_in_use 3000; then
  echo "Port 3000 is already in use."
  kill_port_process 3000
fi

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
  echo "Activating virtual environment from project root..."
  source .venv/bin/activate
elif [ -f "backend/.venv/bin/activate" ]; then
  echo "Activating virtual environment from backend directory..."
  source backend/.venv/bin/activate
else
  echo "No virtual environment found. Please run 'make setup' first."
  exit 1
fi

# Load environment variables
if [ -f ".env" ]; then
  echo "Loading environment variables from .env..."
  source .env
else
  echo "Warning: .env file not found. Ensure environment variables are set."
fi

# Start backend server
echo "🚀 Starting backend API server on http://localhost:8001..."
python3 -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8001 --app-dir . &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 3

# Test backend health
HEALTH_STATUS=$(curl -s http://localhost:8001/health | python3 -c "import json, sys; data = json.load(sys.stdin); print(data.get('status', 'unknown'))" 2>/dev/null)
if [ "$HEALTH_STATUS" = "ok" ]; then
  echo "✅ Backend API is running! API docs available at http://localhost:8001/docs"
else
  echo "❌ Backend API failed to start properly."
  kill $BACKEND_PID 2>/dev/null
  exit 1
fi

# Start frontend server
echo "🚀 Starting frontend server on http://localhost:3000..."
cd frontend && python3 -m http.server 3000 &
FRONTEND_PID=$!

# Wait for frontend to start
echo "⏳ Waiting for frontend to start..."
sleep 3

# Print access information
echo ""
echo "🎉 Phishing Triage System is now running!"
echo "📊 Access the system at:"
echo "   - 💻 Frontend UI: http://localhost:3000/public/test.html"
echo "   - ⚙️ Backend API: http://localhost:8001/docs"
echo ""
echo "Press Ctrl+C to stop all services."

# Wait for user to press Ctrl+C
trap "echo '🛑 Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT
wait

