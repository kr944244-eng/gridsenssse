# ⚡ GridSense – Smart Grid Stability Prediction using Machine Learning

Flask web app + neural network (97.94% accuracy) that predicts whether a smart grid stays stable,
gives a risk level and a model-generated action plan, and includes a chatbot assistant.

**Login:** `admin` / `admin123` (or create an account)

---

## 1. Run on your computer (VS Code)

```bash
pip install -r requirements.txt
python app.py
```
The browser opens at http://127.0.0.1:5000

---

## 2. Why the deployed site showed "404 Page Not Found"

| Cause | Fix |
|---|---|
| The site was put on a **static host** (GitHub Pages, Netlify, Vercel static). These only serve HTML files and **cannot run Python/Flask**, so every page returns 404. | Deploy on a **Python host**: Render (recommended, free), Railway, or Hugging Face Spaces. |
| Only `app.py` / model files were uploaded, and the **`templates/` and `public/` folders were missing**. | Upload the **whole folder** exactly as it is in this zip. |
| No production server / wrong port. | Now included: `gunicorn`, `Procfile`, `render.yaml`, `Dockerfile`, and the app reads the `PORT` variable. |

---

## 3. Deploy on Vercel — step by step

**Why Vercel showed 404 before:** Vercel only runs Flask when `app.py` is at the **root of the repository**
(and `Flask` is in `requirements.txt`). If the files were inside a sub-folder, Vercel built a static site with no pages → 404.

1. Push the project to GitHub so that **`app.py`, `requirements.txt`, `vercel.json`, `models/`, `templates/`, `public/` are at the top level** of the repo.
2. Go to **vercel.com → Add New → Project → Import** your GitHub repo.
3. Settings on the import screen:
   - **Framework Preset:** Flask (auto-detected) — if it shows "Other", pick **Flask**
   - **Root Directory:** `./` (if your repo has a sub-folder, click **Edit** and choose the folder that contains `app.py`)
   - Leave Build / Output / Install commands **empty** (defaults)
   - **Environment Variables:** `SECRET_KEY` = any long random text
4. Click **Deploy** (2–4 min). Open the `https://<project>.vercel.app` link → login page.

Or from the terminal (inside the folder with `app.py`):
```bash
npm i -g vercel
vercel login
vercel          # first deploy (answer the questions, keep defaults)
vercel --prod   # production deploy
```

**Vercel limits to know:** only `/tmp` is writable, so accounts and history are temporary (they reset when Vercel starts a new
instance) — fine for a demo; CSV uploads are limited to 4 MB; the first request after a while takes a few seconds (cold start).

## 3b. Deploy on Render (free) — step by step

1. Create a **GitHub** account → **New repository** (e.g. `gridsense`) → **uploading an existing file**.
2. Drag **everything inside this folder** (app.py, chatbot.py, requirements.txt, Procfile, render.yaml, `models/`,
   `templates/`, `public/` …) and click **Commit changes**.
   *Check on GitHub that you can see the `templates` and `public` folders.*
3. Go to **render.com** → sign in with GitHub → **New + → Web Service** → pick your repo.
4. Fill in:
   - **Runtime:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
   - **Instance type:** Free
5. Under **Environment** add `SECRET_KEY` = any random text, and `DATA_DIR` = `/tmp/gridsense`.
6. Click **Create Web Service**. After the build (~3–5 min) you get a link like `https://gridsense.onrender.com`.

> The free plan sleeps after 15 min of no use – the first visit then takes ~50 s to wake up.
> The free disk is temporary: accounts/history reset when the service restarts (fine for a demo).

**Other hosts:** Railway (uses `Procfile` automatically) · Hugging Face Spaces → *Docker* space (uses `Dockerfile`).

---

## 4. Personalise
- Edit the `PROJECT` section at the top of `app.py` (team names, guide, department).
- Put your college logo at `public/static/img/college_logo.png`.

## 5. Chatbot
The **💬 Ask GridSense** button (every page) answers questions using your own results and the model:
*"What is my last result?"*, *"Why is my grid unstable?"*, *"How do I make it stable?"*,
*"What if industrial reaction time is 2?"*, *"How accurate is the model?"*, *"What is tau?"* …
It works offline. Optional: set the environment variable `ANTHROPIC_API_KEY` so it can also answer free-form questions.

## 6. Files
```
app.py            Flask back end (login, dashboard, prediction API, history, batch, PDF, chat API, error pages)
chatbot.py        GridSense Assistant (answers + what-if predictions)
train.py          Re-trains the models (needs smart_grid_stability_augmented.csv)
models/           Trained neural network (model.pkl) + metrics.json
templates/        HTML pages     public/static/  CSS, JS, sample CSV, logo
requirements.txt  Pinned library versions     Procfile / render.yaml / Dockerfile  deployment configs
```
