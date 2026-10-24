# 🎓 Smart College Attendance System

An AI-powered attendance management system using **face recognition** and **liveness detection**, built with Flask and InsightFace.

---

## ✨ Features

- 👤 **Face Registration** — Enroll students with face photos
- 🔍 **Live Face Recognition** — Real-time attendance via webcam
- 🧠 **Liveness Detection** — Anti-spoofing (rejects photos/videos)
- 📊 **Dashboard & Reports** — Attendance analytics per subject
- 📅 **Timetable Management** — Subject-wise schedule
- 📤 **CSV / DOCX Export** — Download attendance reports
- 🔐 **Auth** — Login-protected admin interface

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12 / Flask 3.1 |
| Face Recognition | InsightFace (ArcFace) |
| Anti-Spoofing | Silent-Face ONNX models |
| Computer Vision | OpenCV |
| Database | SQLite |
| Server | Gunicorn |

---

## 🚀 Local Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd Smart_College_Attendence_system

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Open `http://localhost:5000`

---

## ☁️ Deploy to Railway

### One-time setup

1. Push code to GitHub
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub**
3. Select your repository
4. Railway auto-detects the `nixpacks.toml` and deploys

### Environment Variables (set in Railway dashboard)

| Variable | Description | Example |
|---|---|---|
| `AI_ATTEND_SECRET` | Flask secret key | `your-random-secret-key` |
| `PORT` | Set automatically by Railway | (auto) |

### Persistent Storage (important!)

> ⚠️ Railway's filesystem resets on redeploy. Add a **Volume** for persistent data:
>
> In Railway → your service → **Volumes** → Add Volume → mount at `/app/data`

---

## ⚠️ Camera Note

The **live webcam attendance** feature uses a physical camera.  
On Railway (cloud), the webcam stream uses the **browser's camera** — make sure your browser grants camera permission when accessing the live attendance page.

---

## 📁 Project Structure

```
├── app.py                    # Main Flask application
├── recognition.py            # Face recognition engine
├── register.py               # Student face enrollment
├── liveness.py               # Anti-spoofing module
├── attendance_intelligence.py # AI analytics
├── liveness_models/          # ONNX anti-spoofing models
├── templates/                # HTML pages
├── static/css/               # Stylesheets
├── data/                     # SQLite database (auto-created)
├── students/                 # Enrolled face images
├── requirements.txt
├── Procfile
└── railway.json
```

---

## 🔑 Default Login

Set your admin credentials via the registration page on first run.
