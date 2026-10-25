"""
GridSense - Smart Grid Stability Prediction (Flask back end)
============================================================
Run locally:   python app.py            ->  http://127.0.0.1:5000
Deploy:        gunicorn app:app         (Render / Railway / Hugging Face / any Python host)
First login:   admin / admin123         (or create your own account)

Needs the trained model in models/  (run  python train.py  if it is missing).
"""
import csv
import io
import json
import os
import sqlite3
import threading
import uuid
import webbrowser
from datetime import datetime, timedelta, timezone
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from flask import (Flask, Response, flash, g, jsonify, redirect, render_template,
                   request, send_file, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import chatbot

# =====================================================================
#  EDIT THESE DETAILS FOR YOUR PROJECT (shown in the banner and PDF)
# =====================================================================
PROJECT = {
    "college": "SRM Institute of Science and Technology, Tiruchirappalli",
    "department": "Department of Computer Science and Engineering",
    "title": "Smart Grid Stability Prediction using Machine Learning",
    "team": ["Kishor", "Team Member 2", "Team Member 3"],
    "guide": "Guide: Dr. Faculty Name",
    "app_name": "GridSense",
}
# On a server, set these as environment variables instead of editing the code
SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-to-any-random-text")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
# =====================================================================

BASE = os.path.dirname(os.path.abspath(__file__))
ON_VERCEL = bool(os.environ.get("VERCEL"))             # Vercel sets VERCEL=1 automatically
# CSS / JS / images live in public/static: Vercel serves public/** from its CDN, Flask serves it locally
STATIC_DIR = os.path.join(BASE, "public", "static")
# Database + batch results must be in a writable folder (on Vercel only /tmp is writable)
DATA_DIR = os.environ.get("DATA_DIR", "/tmp/gridsense" if ON_VERCEL else BASE)
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "gridsense.db")
RESULTS_DIR = os.path.join(DATA_DIR, "batch_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

if not os.path.exists(os.path.join(BASE, "models", "model.pkl")):
    raise SystemExit("Model not found. Run  python train.py  first.")
model = joblib.load(os.path.join(BASE, "models", "model.pkl"))
with open(os.path.join(BASE, "models", "metrics.json")) as fh:
    META = json.load(fh)
FEATURES = META["features"]
RANGES = META["ranges"]

# Real-world meaning of each node in the 4-node star grid
NODES = [
    {"id": 1, "key": "plant", "name": "Power Plant", "role": "Producer", "icon": "🏭",
     "desc": "The generating station that supplies power to the whole feeder.",
     "params": ["tau1", "g1"]},
    {"id": 2, "key": "homes", "name": "Residential Area", "role": "Consumer", "icon": "🏘️",
     "desc": "Houses and apartments: lights, fans, ACs and appliances.",
     "params": ["tau2", "p2", "g2"]},
    {"id": 3, "key": "industry", "name": "Industrial Zone", "role": "Consumer", "icon": "🏗️",
     "desc": "Factories and workshops with heavy motors and machines.",
     "params": ["tau3", "p3", "g3"]},
    {"id": 4, "key": "commerce", "name": "Commercial Hub", "role": "Consumer", "icon": "🏬",
     "desc": "Malls, offices, hospitals and shops.",
     "params": ["tau4", "p4", "g4"]},
]
NODE_OF = {p: n for n in NODES for p in n["params"]}

app = Flask(__name__, template_folder=os.path.join(BASE, "templates"),
            static_folder=STATIC_DIR, static_url_path="/static")
app.secret_key = SECRET_KEY
# Vercel accepts request bodies up to 4.5 MB, other hosts 10 MB
app.config["MAX_CONTENT_LENGTH"] = (4 if ON_VERCEL else 10) * 1024 * 1024


# ----------------------------------------------------------------- database
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
        full_name TEXT, password_hash TEXT NOT NULL, created_at TEXT);
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, created_at TEXT,
        site TEXT, inputs TEXT, probability REAL, label TEXT, risk TEXT, recs TEXT);
    CREATE TABLE IF NOT EXISTS batches(
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, created_at TEXT,
        filename TEXT, total INTEGER, stable INTEGER, unstable INTEGER,
        accuracy REAL, result_file TEXT);
    """)
    if not db.execute("SELECT 1 FROM users WHERE username='admin'").fetchone():
        db.execute("INSERT INTO users(username, full_name, password_hash, created_at) VALUES(?,?,?,?)",
                   ("admin", "Grid Operator", generate_password_hash(ADMIN_PASSWORD), now()))
    db.commit()
    db.close()


# Times are stored in Indian Standard Time, even when the server runs in UTC (Render, Railway ...)
APP_TZ = timezone(timedelta(minutes=int(os.environ.get("UTC_OFFSET_MINUTES", 330))))


def now():
    return datetime.now(APP_TZ).strftime("%Y-%m-%d %H:%M:%S")


def time_ago(stamp):
    try:
        then = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=APP_TZ)
    except (TypeError, ValueError):
        return stamp
    s = (datetime.now(APP_TZ) - then).total_seconds()
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{int(s // 60)} min ago"
    if s < 86400:
        return f"{int(s // 3600)} h ago"
    return f"{int(s // 86400)} d ago"


# ----------------------------------------------------------------- auth
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"error": "login required"}), 401
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


@app.context_processor
def inject_globals():
    logo = None
    for name in ("college_logo.png", "college_logo.jpg", "college_logo.jpeg"):
        if os.path.exists(os.path.join(STATIC_DIR, "img", name)):
            logo = "img/" + name
            break
    return {"P": PROJECT, "college_logo": logo, "user_name": session.get("full_name"),
            "username": session.get("username")}


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session and request.method == "GET":
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        user = get_db().execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session.update(user_id=user["id"], username=user["username"],
                           full_name=user["full_name"] or user["username"])
            nxt = request.args.get("next", "")
            return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("dashboard"))
        flash("Wrong username or password.", "error")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        if len(username) < 3 or not username.isalnum():
            flash("Username must be at least 3 letters/numbers (no spaces).", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        elif password != request.form.get("confirm", ""):
            flash("Passwords do not match.", "error")
        else:
            try:
                db = get_db()
                db.execute("INSERT INTO users(username, full_name, password_hash, created_at) VALUES(?,?,?,?)",
                           (username, full_name or username, generate_password_hash(password), now()))
                db.commit()
                flash("Account created. Please log in.", "ok")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("That username is already taken.", "error")
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "ok")
    return redirect(url_for("login"))


# ----------------------------------------------------------------- ML helpers
def clean_inputs(data):
    """Read the 11 model inputs and keep them inside the range the model was trained on."""
    row = {}
    for f in FEATURES:
        lo, hi = RANGES[f]
        try:
            v = float(data.get(f, (lo + hi) / 2))
        except (TypeError, ValueError):
            v = (lo + hi) / 2
        row[f] = float(np.clip(v, lo, hi))
    return row


def risk_level(prob):
    if prob >= 0.8:
        return {"level": "Safe", "color": "#22c55e",
                "meaning": "Normal operation. The grid absorbs price and demand changes and settles quickly.",
                "people": "Homes, industries and hospitals get steady, uninterrupted power."}
    if prob >= 0.5:
        return {"level": "Watch", "color": "#eab308",
                "meaning": "Stable for now, but close to the limit. A small change could tip it over.",
                "people": "Supply is fine today. Operators should keep a close eye on this feeder."}
    if prob >= 0.2:
        return {"level": "At Risk", "color": "#f97316",
                "meaning": "The grid is likely to start oscillating. Preventive action is needed.",
                "people": "Voltage and frequency swings can damage appliances and cause scheduled load-shedding."}
    return {"level": "Critical", "color": "#ef4444",
            "meaning": "Oscillations will grow. Protective relays may trip and cut supply.",
            "people": "High chance of power cuts or blackouts for the whole area. Act immediately."}


def action_text(f, cur, new):
    node = NODE_OF[f]["name"]
    if f.startswith("tau"):
        how = ("Tune the generator's automatic control (AGC / governor) to respond faster."
               if f == "tau1" else
               "Enable automated demand response: smart meters and IoT controllers that react to price signals instantly.")
        verb = "Speed up" if new < cur else "Slow down"
        return f"{verb} response at {node}", f"Reaction time {cur:.1f} s → {new:.1f} s", how
    if f.startswith("g"):
        verb = "Increase" if new > cur else "Reduce"
        how = ("Make tariffs more responsive (time-of-day pricing, incentives) so usage follows price more closely."
               if new > cur else
               "Soften price signals so usage does not over-react and swing back and forth.")
        return f"{verb} price sensitivity at {node}", f"Price sensitivity {cur:.2f} → {new:.2f}", how
    # power demand (stored as negative numbers in the dataset)
    verb = "Reduce" if abs(new) < abs(cur) else "Increase"
    how = ("Shift heavy loads (water pumps, EV charging, ACs, furnaces) to off-peak hours."
           if verb == "Reduce" else
           "Schedule flexible loads into this period to balance the network.")
    return f"{verb} demand at {node}", f"Demand {abs(cur):.2f} → {abs(new):.2f} p.u.", how


def _logit(p):
    p = np.clip(p, 1e-300, 1 - 1e-16)
    return np.log(p) - np.log1p(-p)


def recommend(row, prob, goal=0.95, max_steps=4):
    """Build an action plan with the model itself.

    Greedy search: at each step try every reading across its range (with the earlier steps
    already applied), keep the smallest change that gives (almost) the biggest improvement,
    and repeat until the grid is safely stable. Improvement is measured in log-odds, so even
    grids at ~0% get useful first steps.
    """
    if prob >= goal:
        return [], None
    cur, cur_p, used, plan, steps = dict(row), prob, set(), [], 30
    for _ in range(max_steps):
        feats = [f for f in FEATURES if f not in used]
        blocks = []
        for f in feats:
            blk = pd.DataFrame([cur] * steps)[FEATURES]
            blk[f] = np.linspace(*RANGES[f], steps)
            blocks.append(blk)
        allp = model.predict_proba(pd.concat(blocks, ignore_index=True))[:, 1]
        base = _logit(cur_p)
        reach, improve = [], []
        for i, f in enumerate(feats):
            xs = blocks[i][f].to_numpy()
            ps = allp[i * steps:(i + 1) * steps]
            span = RANGES[f][1] - RANGES[f][0]
            ok = np.where(ps >= goal)[0]
            if len(ok):                                       # this change alone makes the grid safe
                j = ok[np.argmin(np.abs(xs[ok] - cur[f]))]    # ...pick the smallest such change
                reach.append((abs(xs[j] - cur[f]) / span, f, float(xs[j]), float(ps[j])))
                continue
            gains = _logit(ps) - base
            if gains.max() > 0.3:                             # otherwise: biggest move towards safety,
                ok = np.where(gains >= 0.8 * gains.max())[0]  # without going to extremes
                j = ok[np.argmin(np.abs(xs[ok] - cur[f]))]
                improve.append((gains[j], f, float(xs[j]), float(ps[j])))
        if reach:
            _, f, new, p = min(reach)
        elif improve:
            _, f, new, p = max(improve)
        else:
            break
        title, change, how = action_text(f, cur[f], new)
        plan.append({"feature": f, "title": title, "change": change, "how": how,
                     "node": NODE_OF[f]["name"], "icon": NODE_OF[f]["icon"],
                     "new_value": round(new, 3), "prob_after": round(p, 4)})
        cur[f], cur_p = new, p
        used.add(f)
        if cur_p >= goal:
            break
    return plan, (round(cur_p, 4) if plan else None)


# ----------------------------------------------------------------- pages
@app.route("/")
@login_required
def dashboard():
    db = get_db()
    preds = db.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC",
                       (session["user_id"],)).fetchall()
    batches = db.execute("SELECT COUNT(*) n, COALESCE(SUM(total),0) grids, COALESCE(SUM(unstable),0) bad "
                         "FROM batches WHERE user_id=?", (session["user_id"],)).fetchone()
    total = len(preds)
    stable = sum(1 for p in preds if p["label"] == "stable")
    risk_counts = {k: sum(1 for p in preds if p["risk"] == k) for k in ("Safe", "Watch", "At Risk", "Critical")}
    trend = [{"id": p["id"], "t": p["created_at"], "site": p["site"], "p": round(p["probability"] * 100, 1),
              "risk": p["risk"]} for p in reversed(preds[:20])]
    alerts = []
    for p in preds:
        if p["risk"] in ("Critical", "At Risk") and len(alerts) < 4:
            recs = json.loads(p["recs"] or "{}").get("recs", [])
            alerts.append({"id": p["id"], "site": p["site"], "time": p["created_at"], "risk": p["risk"],
                           "prob": p["probability"], "action": recs[0]["title"] if recs else None})
    best = next(r for r in META["results"] if r["model"] == META["best_model"])
    data = {
        "total": total, "stable": stable, "unstable": total - stable,
        "health": (stable / total * 100) if total else None,
        "avg": (sum(p["probability"] for p in preds) / total * 100) if total else None,
        "critical": risk_counts["Critical"] + risk_counts["At Risk"],
        "risk": risk_counts, "trend": trend, "alerts": alerts,
        "batches": dict(batches), "best": best, "best_model": META["best_model"],
        "importance": META["importance"][:5], "dataset": META["dataset"],
    }
    # activity timeline: latest single checks + batch uploads, newest first
    acts = [{"kind": "check", "t": p["created_at"], "site": p["site"], "label": p["label"], "risk": p["risk"],
             "prob": p["probability"], "id": p["id"]} for p in preds[:8]]
    for bt in db.execute("SELECT * FROM batches WHERE user_id=? ORDER BY id DESC LIMIT 5", (session["user_id"],)):
        acts.append({"kind": "batch", "t": bt["created_at"], "site": bt["filename"], "total": bt["total"],
                     "unstable": bt["unstable"], "id": bt["id"]})
    data["activity"] = sorted(acts, key=lambda a: a["t"], reverse=True)[:7]
    for a in data["activity"]:
        a["ago"] = time_ago(a["t"])
    data["latest"] = ({"label": preds[0]["label"], "site": preds[0]["site"], "prob": preds[0]["probability"]}
                      if preds else None)
    return render_template("dashboard.html", d=data, recent=preds[:6], active="dashboard")


@app.route("/api/quick", methods=["POST"])
@login_required
def api_quick():
    """Dashboard quick check: 4 reaction times + one shared price sensitivity, typical demand. Not saved."""
    body = request.get_json(force=True) or {}
    g = body.get("g", 0.5)
    raw = {f"tau{i}": body.get(f"tau{i}", 5.0) for i in range(1, 5)}
    raw.update({f"g{i}": g for i in range(1, 5)})
    raw.update({f"p{i}": -1.25 for i in range(2, 5)})
    row = clean_inputs(raw)
    prob = float(model.predict_proba(pd.DataFrame([row])[FEATURES])[0][1])
    r = risk_level(prob)
    return jsonify({"probability_stable": round(prob, 4), "label": "stable" if prob >= 0.5 else "unstable",
                    "risk": r["level"], "color": r["color"], "meaning": r["meaning"]})


@app.route("/predict")
@login_required
def predict_page():
    return render_template("predict.html", nodes=NODES, ranges=RANGES, active="predict")


@app.route("/history")
@login_required
def history_page():
    db = get_db()
    preds = db.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC",
                       (session["user_id"],)).fetchall()
    batches = db.execute("SELECT * FROM batches WHERE user_id=? ORDER BY id DESC",
                         (session["user_id"],)).fetchall()
    total = len(preds)
    stable = sum(1 for p in preds if p["label"] == "stable")
    avg = (sum(p["probability"] for p in preds) / total) if total else 0
    return render_template("history.html", preds=preds, batches=batches, active="history",
                           stats={"total": total, "stable": stable, "unstable": total - stable, "avg": avg})


@app.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    db = get_db()
    db.execute("DELETE FROM predictions WHERE user_id=?", (session["user_id"],))
    db.execute("DELETE FROM batches WHERE user_id=?", (session["user_id"],))
    db.commit()
    flash("History cleared.", "ok")
    return redirect(url_for("history_page"))


@app.route("/history/export.csv")
@login_required
def export_history():
    preds = get_db().execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id",
                             (session["user_id"],)).fetchall()
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["id", "time", "site"] + FEATURES + ["probability_stable", "result", "risk"])
    for p in preds:
        inp = json.loads(p["inputs"])
        w.writerow([p["id"], p["created_at"], p["site"]] + [inp[f] for f in FEATURES] +
                   [p["probability"], p["label"], p["risk"]])
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=prediction_history.csv"})


@app.route("/insights")
@login_required
def insights_page():
    return render_template("insights.html", active="insights")


@app.route("/batch", methods=["GET", "POST"])
@login_required
def batch_page():
    result = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename.lower().endswith(".csv"):
            flash("Please choose a .csv file.", "error")
            return redirect(url_for("batch_page"))
        try:
            df = pd.read_csv(file)
        except Exception:
            flash("Could not read that file. Make sure it is a valid CSV.", "error")
            return redirect(url_for("batch_page"))
        df.columns = [c.strip().lower() for c in df.columns]
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            flash("Missing columns: " + ", ".join(missing) + ". Download the sample file to see the format.", "error")
            return redirect(url_for("batch_page"))
        X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
        bad = X.isna().any(axis=1)
        X, df = X[~bad], df[~bad].copy()
        if len(X) == 0:
            flash("No valid rows found in the file.", "error")
            return redirect(url_for("batch_page"))
        if len(X) > 100000:
            X, df = X.iloc[:100000], df.iloc[:100000]
        for p in ("p2", "p3", "p4"):                       # accept demand typed as positive numbers
            if (X[p] > 0).all():
                X[p] = -X[p]
        out_of_range = int(sum(((X[f] < RANGES[f][0] - 1e-6) | (X[f] > RANGES[f][1] + 1e-6)).sum()
                               for f in FEATURES))
        for f in FEATURES:
            X[f] = X[f].clip(*RANGES[f])
        prob = model.predict_proba(X)[:, 1]
        df["probability_stable"] = prob.round(4)
        df["prediction"] = np.where(prob >= 0.5, "stable", "unstable")
        df["risk"] = [risk_level(p)["level"] for p in prob]
        accuracy = None
        if "stabf" in df.columns:
            accuracy = float((df["stabf"].astype(str).str.strip().str.lower() == df["prediction"]).mean())
        fname = f"{uuid.uuid4().hex}.csv"
        df.to_csv(os.path.join(RESULTS_DIR, fname), index=False)
        stable = int((df["prediction"] == "stable").sum())
        db = get_db()
        cur = db.execute("INSERT INTO batches(user_id, created_at, filename, total, stable, unstable, accuracy, result_file)"
                         " VALUES(?,?,?,?,?,?,?,?)",
                         (session["user_id"], now(), file.filename, len(df), stable, len(df) - stable, accuracy, fname))
        db.commit()
        risk_counts = df["risk"].value_counts().to_dict()
        result = {"id": cur.lastrowid, "filename": file.filename, "total": len(df), "stable": stable,
                  "unstable": len(df) - stable, "accuracy": accuracy, "skipped": int(bad.sum()),
                  "out_of_range": out_of_range,
                  "risk": {k: int(risk_counts.get(k, 0)) for k in ("Safe", "Watch", "At Risk", "Critical")},
                  "columns": FEATURES + ["probability_stable", "prediction", "risk"] + (["stabf"] if accuracy is not None else []),
                  "rows": df.head(200).to_dict("records")}
    return render_template("batch.html", result=result, active="batch")


@app.route("/batch/download/<int:batch_id>")
@login_required
def batch_download(batch_id):
    b = get_db().execute("SELECT * FROM batches WHERE id=? AND user_id=?", (batch_id, session["user_id"])).fetchone()
    if not b or not os.path.exists(os.path.join(RESULTS_DIR, b["result_file"])):
        flash("That result file is no longer available.", "error")
        return redirect(url_for("history_page"))
    return send_file(os.path.join(RESULTS_DIR, b["result_file"]), as_attachment=True,
                     download_name="predictions_" + os.path.splitext(b["filename"])[0] + ".csv")


# ----------------------------------------------------------------- API
@app.route("/api/meta")
@login_required
def api_meta():
    return jsonify(META)


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    body = request.get_json(force=True)
    row = clean_inputs(body.get("inputs", {}))
    site = (body.get("site") or "Feeder 1").strip()[:80]
    prob = float(model.predict_proba(pd.DataFrame([row])[FEATURES])[0][1])
    label = "stable" if prob >= 0.5 else "unstable"
    risk = risk_level(prob)
    recs, combined = recommend(row, prob)
    db = get_db()
    cur = db.execute("INSERT INTO predictions(user_id, created_at, site, inputs, probability, label, risk, recs)"
                     " VALUES(?,?,?,?,?,?,?,?)",
                     (session["user_id"], now(), site, json.dumps(row), prob, label, risk["level"],
                      json.dumps({"recs": recs, "combined": combined})))
    db.commit()
    return jsonify({"id": cur.lastrowid, "probability_stable": round(prob, 4), "label": label,
                    "risk": risk, "recommendations": recs, "combined_after": combined,
                    "p1": round(-(row["p2"] + row["p3"] + row["p4"]), 3), "inputs": row})


@app.route("/api/sweep", methods=["POST"])
@login_required
def api_sweep():
    body = request.get_json(force=True)
    param = body.get("param", "tau1")
    if param not in FEATURES:
        return jsonify({"error": "unknown parameter"}), 400
    base = clean_inputs(body.get("inputs", {}))
    lo, hi = RANGES[param]
    xs = np.linspace(lo, hi, 50)
    grid = pd.DataFrame([base] * len(xs))[FEATURES]
    grid[param] = xs
    probs = model.predict_proba(grid)[:, 1]
    return jsonify({"param": param, "x": xs.round(3).tolist(), "y": probs.round(4).tolist()})


# ----------------------------------------------------------------- chatbot
def _user_stats(db, uid):
    preds = db.execute("SELECT label, probability, risk FROM predictions WHERE user_id=?", (uid,)).fetchall()
    b = db.execute("SELECT COUNT(*) n, COALESCE(SUM(total),0) g, COALESCE(SUM(unstable),0) u FROM batches WHERE user_id=?",
                   (uid,)).fetchone()
    total = len(preds)
    stable = sum(1 for p in preds if p["label"] == "stable")
    return {"total": total, "stable": stable, "unstable": total - stable,
            "avg": (sum(p["probability"] for p in preds) / total) if total else 0,
            "risk": {k: sum(1 for p in preds if p["risk"] == k) for k in ("Safe", "Watch", "At Risk", "Critical")},
            "batches": b["n"], "batch_grids": b["g"], "batch_bad": b["u"]}


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    msg = str((request.get_json(silent=True) or {}).get("message", ""))[:500]
    db = get_db()
    row = db.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 1",
                     (session["user_id"],)).fetchone()
    last = None
    if row:
        extra = json.loads(row["recs"] or "{}")
        last = {"id": row["id"], "site": row["site"], "created_at": row["created_at"],
                "inputs": json.loads(row["inputs"]), "probability": row["probability"], "label": row["label"],
                "plan": extra.get("recs", []), "combined": extra.get("combined")}
    chat_state = session.get("chat", {})
    ctx = {"meta": META, "nodes": NODES, "ranges": RANGES, "last": last, "user": session.get("full_name"),
           "stats": _user_stats(db, session["user_id"]), "risk_level": risk_level, "session": chat_state,
           "predict": lambda r: float(model.predict_proba(pd.DataFrame([clean_inputs(r)])[FEATURES])[0][1]),
           "recommend": recommend}
    try:
        out = chatbot.answer(msg, ctx)
    except Exception:
        app.logger.exception("chat failed")
        out = {"reply": "Sorry, something went wrong while answering. Please try asking in a different way.", "chips": []}
    session["chat"] = chat_state
    return jsonify(out)


# ----------------------------------------------------------------- health + errors
@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "model": META["best_model"]})


@app.errorhandler(404)
def not_found(_e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "not found"}), 404
    return render_template("error.html", code=404, title="Page not found",
                           msg="The page you are looking for does not exist or was moved."), 404


@app.errorhandler(500)
def server_error(_e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "server error"}), 500
    return render_template("error.html", code=500, title="Something went wrong",
                           msg="The server hit an unexpected error. Please try again."), 500


@app.errorhandler(413)
def too_large(_e):
    flash("That file is too big. The limit is 10 MB.", "error")
    return redirect(url_for("batch_page"))


# ----------------------------------------------------------------- PDF report
@app.route("/report/<int:pred_id>.pdf")
@login_required
def report_pdf(pred_id):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    p = get_db().execute("SELECT * FROM predictions WHERE id=? AND user_id=?",
                         (pred_id, session["user_id"])).fetchone()
    if not p:
        flash("Report not found.", "error")
        return redirect(url_for("history_page"))
    inp = json.loads(p["inputs"])
    extra = json.loads(p["recs"] or "{}")
    risk = risk_level(p["probability"])
    best = next(r for r in META["results"] if r["model"] == META["best_model"])

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title=f"Grid Stability Report #{p['id']}")
    ss = getSampleStyleSheet()
    H = ParagraphStyle("H", parent=ss["Title"], fontSize=17, leading=21, spaceAfter=2, textColor=colors.HexColor("#0b2a4a"))
    S = ParagraphStyle("S", parent=ss["Normal"], fontSize=9.5, leading=13, textColor=colors.HexColor("#334155"))
    B = ParagraphStyle("B", parent=ss["Normal"], fontSize=10, leading=14)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#0b2a4a"))
    accent = colors.HexColor(risk["color"])
    story = []

    head_text = Paragraph(f"<b>{PROJECT['college']}</b><br/>{PROJECT['department']}<br/>"
                          f"<font color='#0891b2'><b>{PROJECT['title']}</b></font>", S)
    logo_path = next((os.path.join(STATIC_DIR, "img", n) for n in
                      ("college_logo.png", "college_logo.jpg", "college_logo.jpeg")
                      if os.path.exists(os.path.join(STATIC_DIR, "img", n))), None)
    if logo_path:
        head = Table([[Image(logo_path, width=22 * mm, height=22 * mm, kind="proportional"), head_text]],
                     colWidths=[26 * mm, None])
    else:
        head = Table([[head_text]])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("LINEBELOW", (0, 0), (-1, -1), 1.2, colors.HexColor("#0891b2")),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [head, Spacer(1, 10), Paragraph("Grid Stability Assessment Report", H)]
    story.append(Paragraph(f"Report #{p['id']} &nbsp;·&nbsp; Site: <b>{p['site']}</b> &nbsp;·&nbsp; "
                           f"Date: {p['created_at']} &nbsp;·&nbsp; Prepared by: {session.get('full_name')}", S))
    story.append(Spacer(1, 10))

    verdict = Table([[Paragraph(f"<font size=15 color='{risk['color']}'><b>GRID {p['label'].upper()}</b></font>", B),
                      Paragraph(f"<font size=15><b>{p['probability'] * 100:.1f}%</b></font><br/>probability stable", S),
                      Paragraph(f"<font size=13 color='{risk['color']}'><b>{risk['level']}</b></font><br/>risk level", S)]],
                    colWidths=[70 * mm, 50 * mm, None])
    verdict.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1.5, accent),
                                 ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                 ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
    story += [verdict, Spacer(1, 6),
              Paragraph(f"<b>What this means:</b> {risk['meaning']}", B),
              Paragraph(f"<b>Impact on people:</b> {risk['people']}", B)]

    story.append(Paragraph("Readings entered", H2))
    rows = [["Location", "Role", "Reaction time (s)", "Power (p.u.)", "Price sensitivity"]]
    p1 = -(inp["p2"] + inp["p3"] + inp["p4"])
    for n in NODES:
        i = n["id"]
        power = f"{p1:.3f} (supplied)" if i == 1 else f"{abs(inp[f'p{i}']):.3f} (demand)"
        rows.append([n["name"], n["role"], f"{inp[f'tau{i}']:.3f}", power, f"{inp[f'g{i}']:.3f}"])
    t = Table(rows, colWidths=[38 * mm, 24 * mm, 34 * mm, 36 * mm, None])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b2a4a")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.append(t)

    story.append(Paragraph("Recommended actions", H2))
    recs = extra.get("recs", [])
    if recs:
        for k, r in enumerate(recs, 1):
            story.append(Paragraph(f"<b>Step {k}. {r['title']}</b> ({r['change']}): stability after this step "
                                   f"<b>{'&lt;0.1' if 0 < r['prob_after'] < 0.001 else format(r['prob_after'] * 100, '.1f')}%</b>.<br/><font color='#475569'>{r['how']}</font>", B))
            story.append(Spacer(1, 4))
        if extra.get("combined") is not None:
            story.append(Paragraph(f"After following all the steps above, the predicted stability is "
                                   f"<b>{extra['combined'] * 100:.1f}%</b>.", B))
    else:
        story.append(Paragraph("No change is needed. The grid is already operating with a healthy safety margin. "
                               "Continue routine monitoring.", B))

    story.append(Paragraph("About the prediction", H2))
    story.append(Paragraph(
        f"Model: <b>{META['best_model']}</b>, trained on {META['dataset']['train_rows']:,} simulated grid scenarios "
        f"and tested on {META['dataset']['test_rows']:,} unseen ones. Test accuracy {best['accuracy'] * 100:.2f}%, "
        f"F1 score {best['f1']:.3f}, ROC-AUC {best['roc_auc']:.4f}. Data: Decentral Smart Grid Control (DSGC) "
        f"4-node star grid simulations (UCI / Kaggle).", S))
    story.append(Spacer(1, 16))
    story.append(Paragraph(f"Generated by {PROJECT['app_name']} · {', '.join(PROJECT['team'])} · {PROJECT['guide']}", S))

    doc.build(story)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True,
                     download_name=f"grid_report_{p['id']}.pdf")


# ----------------------------------------------------------------- run
init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    on_server = "PORT" in os.environ                   # hosting platforms set PORT
    host = "0.0.0.0" if on_server else "127.0.0.1"
    url = f"http://127.0.0.1:{port}"
    print(f"\n  {PROJECT['app_name']} is running at {url}")
    print("  Login: admin / admin123  (or create your own account)")
    print("  Press CTRL+C to stop.\n")
    if not on_server:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=False)
