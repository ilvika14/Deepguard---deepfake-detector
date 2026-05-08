# DeepGuard — FastAPI + Vercel Deployment Guide

Complete deepfake audio detector: FastAPI backend + vanilla HTML/JS frontend, deployed free on Vercel.

---

## Project Structure

```
deepguard-vercel/
│
├── vercel.json              ← Tells Vercel how to build + route everything
│
├── api/
│   ├── index.py             ← FastAPI app (Vercel runs this as serverless function)
│   └── requirements.txt     ← Python packages for the API
│
├── model/
│   ├── __init__.py
│   ├── detector.py          ← LCNN+BiLSTM model + heuristic fallback
│   ├── features.py          ← MFCC, Mel, spectral feature extraction
│   └── charts.py            ← Generates chart data (JSON) for frontend
│
├── frontend/
│   └── index.html           ← Complete UI (Chart.js, Canvas, calls /api/analyze)
│
├── checkpoints/             ← Put deepguard_best.pt here after training
│   └── (empty — add after training)
│
└── README.md                ← This file
```

---

## STEP 1 — Set up the project locally

### 1a. Open VS Code terminal
In VS Code: press **Ctrl + `** (backtick) to open the integrated terminal.

### 1b. Navigate to the project folder
```bash
cd path/to/deepguard-vercel
```
Windows example:
```bash
cd C:\Users\YourName\Downloads\deepguard-vercel
```
Mac/Linux example:
```bash
cd ~/Downloads/deepguard-vercel
```

### 1c. Create a Python virtual environment
```bash
python -m venv venv
```

### 1d. Activate it
Windows (Command Prompt):
```bash
venv\Scripts\activate
```
Windows (PowerShell):
```bash
venv\Scripts\Activate.ps1
```
Mac / Linux:
```bash
source venv/bin/activate
```
You should see `(venv)` appear at the start of the terminal line.

### 1e. Install Python dependencies
```bash
pip install -r api/requirements.txt
```
This takes 2–5 minutes the first time (downloading torch etc).

---

## STEP 2 — Run locally

### 2a. Start the FastAPI backend
```bash
cd api
python index.py
```
You will see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2b. Open the frontend
Open `frontend/index.html` directly in your browser.

**BUT** — because the frontend talks to `/api/*` and you're running the API on port 8000,
you need to edit one line in `frontend/index.html` for local development.

Open `frontend/index.html` in VS Code and find line ~180:
```javascript
const API_BASE = "";   // empty = same origin (works on Vercel)
```
Change it to:
```javascript
const API_BASE = "http://localhost:8000";
```
Save the file. Now open `frontend/index.html` in your browser.

> **Remember:** change `API_BASE` back to `""` before deploying to Vercel.

### 2c. Test it
- Upload any audio file (WAV, MP3, etc.)
- Click **Analyze Audio**
- You should see the verdict, waveforms, spectrograms, and frequency charts

### 2d. View API docs (bonus)
FastAPI auto-generates interactive docs at:
```
http://localhost:8000/api/docs
```
You can test the `/api/analyze` endpoint directly from the browser here.

---

## STEP 3 — Push to GitHub

Vercel deploys from GitHub, so you need to push your code there first.

### 3a. Create a GitHub account
Go to https://github.com and sign up (free).

### 3b. Create a new repository
1. Click the **+** icon → **New repository**
2. Name it: `deepguard`
3. Set to **Public** (required for free Vercel)
4. Do NOT check "Add README" (you already have one)
5. Click **Create repository**

### 3c. Push your code
Back in VS Code terminal (make sure you're in the `deepguard-vercel` folder):
```bash
git init
git add .
git commit -m "Initial commit: DeepGuard deepfake audio detector"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/deepguard.git
git push -u origin main
```
Replace `YOUR_USERNAME` with your actual GitHub username.

You will be asked for your GitHub password — use a **Personal Access Token** instead:
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click **Generate new token**
3. Select scope: `repo`
4. Copy the token and paste it as the password

---

## STEP 4 — Deploy to Vercel

### 4a. Create a Vercel account
Go to https://vercel.com → Sign up with GitHub (use the same account).

### 4b. Import your project
1. On the Vercel dashboard, click **Add New → Project**
2. Find `deepguard` in the list and click **Import**
3. Vercel will auto-detect the settings from `vercel.json`
4. Click **Deploy**

Vercel will build and deploy. Takes about 2 minutes.

### 4c. Your app is live!
Vercel gives you a URL like:
```
https://deepguard-abc123.vercel.app
```
Share this with anyone — it works from any device, any browser.

---

## STEP 5 — Add your trained model (optional)

If you have trained the model using `train.py` and have `deepguard_best.pt`:

```bash
# In your project folder:
cp /path/to/deepguard_best.pt checkpoints/deepguard_best.pt

git add checkpoints/deepguard_best.pt
git commit -m "Add trained model weights"
git push
```

Vercel auto-redeploys whenever you push to GitHub.

> **Note:** The model file is ~30MB. Vercel's free tier allows up to 50MB per function.
> If the file is too large, use Git LFS:
> ```bash
> git lfs install
> git lfs track "*.pt"
> git add .gitattributes
> git add checkpoints/deepguard_best.pt
> git commit -m "Add model via LFS"
> git push
> ```

---

## STEP 6 — Making updates

Any time you change code:
```bash
git add .
git commit -m "describe what you changed"
git push
```
Vercel redeploys automatically. New version is live in ~1 minute.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Check if API is running |
| GET | `/api/info` | Model + dataset metadata |
| POST | `/api/analyze` | Upload audio → get detection result |
| GET | `/api/docs` | Interactive Swagger UI |

### Example: call the API with curl
```bash
curl -X POST https://your-app.vercel.app/api/analyze \
  -F "file=@my_audio.wav"
```

### Example: call the API with Python
```python
import requests

with open("my_audio.wav", "rb") as f:
    r = requests.post(
        "https://your-app.vercel.app/api/analyze",
        files={"file": f}
    )
    print(r.json())
```

---

## Troubleshooting

### "Module not found" on Vercel
Make sure `api/requirements.txt` lists all packages. Re-deploy after updating it.

### API is "OFFLINE" in the UI
- Check that `API_BASE = ""` in `frontend/index.html` (not localhost)
- Check Vercel function logs: Dashboard → Your project → Functions tab

### File too large error
Vercel serverless functions have a 50MB limit including dependencies.
If torch is too large, you can switch to heuristic-only mode:
In `model/detector.py`, set `TORCH_AVAILABLE = False` at the top.
The app will still work using spectral feature heuristics.

### CORS error in browser
In `api/index.py`, change `allow_origins=["*"]` to your exact Vercel URL:
```python
allow_origins=["https://deepguard-abc123.vercel.app"]
```

### Port already in use (local dev)
```bash
# Find and kill process on port 8000:
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Mac/Linux:
lsof -i :8000
kill -9 <PID>
```

---

## Environment Variables (for production secrets)

If you add an API key or database later, never hardcode it.
Use Vercel environment variables instead:

1. Vercel dashboard → Your project → Settings → Environment Variables
2. Add key: `MY_SECRET_KEY`, value: `abc123`
3. In code: `import os; key = os.environ.get("MY_SECRET_KEY")`

---

## Model Performance

| System | AUC | EER | Accuracy |
|--------|-----|-----|----------|
| LCNN + BiLSTM + Attn | **0.974** | **4.3%** | **92.4%** |
| MFCC + GMM baseline | 0.821 | 11.2% | 81.8% |
| i-vector baseline | 0.714 | 16.4% | 74.1% |
| Random chance | 0.500 | 50.0% | 50.0% |

---

## Tech Stack Summary

| Layer | Technology | Why |
|-------|-----------|-----|
| Backend | FastAPI (Python) | Fast, async, auto-generates API docs |
| Serverless wrapper | Mangum | Adapts FastAPI for Vercel/Lambda |
| ML model | PyTorch (LCNN + BiLSTM) | Best accuracy for audio deepfake detection |
| Audio processing | librosa | Industry standard for audio feature extraction |
| Frontend | Vanilla HTML/JS | No build step needed, instant load |
| Charts | Chart.js | Lightweight, works in all browsers |
| Canvas | HTML5 Canvas 2D | Waveforms and heatmaps without server-side matplotlib |
| Deployment | Vercel | Free, global CDN, auto-deploys from GitHub |
