# Setup Guide — Helmet Detection

Motorcycle helmet-violation detection: FastAPI + YOLO backend, Next.js frontend.

**Requirements:** Python 3.10+, Node.js 18+, PostgreSQL 14+, webcam or RTSP 

## 1. Backend

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**GPU (optional)** — the app auto-detects CUDA and falls back to CPU. On an
NVIDIA machine, install the CUDA torch build *before* `pip install -r` so it's
kept (see instructions in `backend/requirements.txt`):

```powershell
nvidia-smi                                                        # confirm driver + CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

CPU-only machines need nothing extra — plain `pip install -r requirements.txt`
installs the CPU build.

**Database** — create it, then set credentials in `backend/.env` (see `.env.example`):

```sql
CREATE DATABASE helmet_detection;
```

```env
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_USER=postgres
DATABASE_PASSWORD=<your-password>
DATABASE_NAME=helmet_detection
```

Tables are created automatically on first start — no migrations needed.

**Video source** — pick one:

| Source | Where |
|---|---|
| Video file | `video_path` in `config.development.json` |
| USB webcam | `"use_webcam": true`, `"webcam_id": 0` in the same file |
| RTSP camera | `RTSP_VIDEO_PATH=rtsp://user:pass@ip:554/stream2` in `.env` (overrides both) |

Tip: prefer the camera's **sub-stream** (`stream2`) for detection — it is far
cheaper to decode and keeps stream latency low. The backend holds only ONE
RTSP session shared by all viewers, but cameras limit concurrent sessions, so
don't keep extra VLC/browser streams open.

**Run:**

```powershell
python main.py          # or: uvicorn main:app --reload --port 8000
```

API at `http://localhost:8000` · Swagger at `/docs`

## 2. Frontend

```powershell
cd frontend
npm install
```

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

```powershell
npm run dev             # → http://localhost:3000
```

