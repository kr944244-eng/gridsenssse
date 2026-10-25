"""
GridSense Assistant - chatbot for project and result questions
==============================================================
Works fully offline: it understands common questions with keyword rules and answers from
live data (the user's own predictions in the database) and from the trained model itself.

It can also run "what-if" predictions from a sentence, for example:
    "what if the industrial reaction time is 2"      "set tau3 = 2 and g2 = 0.3"

Optional: set the environment variable ANTHROPIC_API_KEY to let it answer free-form
questions it does not recognise with an LLM (the project data is sent as context).
"""
import json
import os
import re
import urllib.request

NODE_WORDS = [
    (1, r"plant|producer|generat|station"),
    (2, r"residential|home|house|apartment|domestic"),
    (3, r"industr|factor|workshop"),
    (4, r"commercial|mall|office|hospital|shop|business"),
]
KIND_WORDS = [
    ("tau", r"reaction|response|react|delay|tau|τ"),
    ("p", r"demand|power|load|consum"),
    ("g", r"sensitiv|elastic|gamma|γ"),
]
KIND_LABEL = {"tau": "reaction time", "p": "power demand", "g": "price sensitivity"}
NUM = r"(-?\d+(?:\.\d+)?)"

INTENTS = [
    ("greet",    r"^\s*(hi|hello|hey|hai|vanakkam|good (morning|afternoon|evening))\b"),
    ("thanks",   r"\b(thank|thanks|thx|great|awesome|nice|super)\b"),
    ("help",     r"\b(help|what can you|what do you do|commands|options|menu)\b"),
    ("last",     r"\b(last|latest|recent|previous|my)\b.*\b(result|prediction|check|report|grid|output)\b|^\s*(result|status)\s*\??$"),
    ("why",      r"\b(why|reason|cause|explain (the|my) result|what went wrong)\b"),
    ("fix",      r"\b(fix|improve|make (it|the grid|my grid)? ?stable|prevent|recommend|suggest|solution|action plan|actions|what should (i|we) do|how (can|do|should) (i|we) (fix|make|improve|stabili[sz]e|prevent|avoid))\b"),
    ("stats",    r"\b(how many|history|summary|statistics|stats|total|count|overall|so far)\b"),
    ("accuracy", r"\b(accura|precision|recall|f1|auc|roc|performance|how good|confusion|reliable|trust|error)\w*"),
    ("models",   r"\b(compare|comparison|which model|best model|models? used|algorithm|ann|neural|random forest|boosting|logistic|mlp)\b"),
    ("features", r"\b(important|importance|feature|factor|affect|drive|influence|matter)\w*"),
    ("tau",      r"\b(tau|reaction time|response time|delay)\b"),
    ("power",    r"\b(power|demand|p1|p2|p3|p4|load)\b"),
    ("gamma",    r"\b(elasticity|gamma|sensitivity|price sensitivity)\b"),
    ("risk",     r"\b(risk level|risk|safe|watch|critical|at risk)\b"),
    ("dataset",  r"\b(dataset|data set|data|kaggle|uci|rows|columns|records|simulation)\b"),
    ("leakage",  r"\b(leak|leakage|stab\b|drop|split|augment|group)\w*"),
    ("project",  r"\b(smart grid|project|what is this|about|dsgc|purpose|objective|problem)\b"),
    ("pdf",      r"\b(pdf|report)\b"),
    ("usage",    r"\b(batch|csv|upload|how to use|website|page|login|sample file)\b"),
    ("bye",      r"\b(bye|goodbye|see you|exit|quit)\b"),
]

SUGGEST = ["What is my last result?", "Why is my grid unstable?", "How do I make it stable?",
           "What if industrial reaction time is 2?", "Which features matter most?", "Compare the models"]


def pct(p):
    return "<0.1%" if 0 < p < 0.001 else f"{p * 100:.1f}%"


# ------------------------------------------------------------------ parsing what-if changes
def parse_changes(text, ranges):
    """Return {feature: model_value} found in the message, plus notes about clipping."""
    t = text.lower().replace("γ", " gamma ").replace("τ", " tau ")
    changes, notes = {}, []
    # 1) code form: tau3 = 2, g2 to 0.3, p4: 1.5
    for kind, idx, val in re.findall(r"\b(tau|p|g)\s*([1-4])\s*(?:=|:|to|is|becomes|->|as)?\s*" + NUM, t):
        changes[f"{kind}{idx}"] = float(val)
    # 2) natural form, clause by clause: "industrial reaction time to 2"
    if not changes:
        for clause in re.split(r"[,;]|\band\b|\bthen\b", t):
            nums = re.findall(NUM, clause)
            if not nums:
                continue
            node = next((n for n, pat in NODE_WORDS if re.search(pat, clause)), None)
            kind = next((k for k, pat in KIND_WORDS if re.search(pat, clause)), None)
            if node and kind:
                changes[f"{kind}{node}"] = float(nums[-1])
    out = {}
    for f, v in changes.items():
        if f == "p1":
            notes.append("p1 (plant output) is calculated automatically as p2+p3+p4, so I ignored it.")
            continue
        if f not in ranges:
            continue
        if f.startswith("p"):
            v = -abs(v)                                     # demand is stored as a negative number
        lo, hi = ranges[f]
        if v < lo or v > hi:
            shown = (lambda x: abs(x)) if f.startswith("p") else (lambda x: x)
            notes.append(f"{f} = {shown(v):g} is outside the trained range "
                         f"({min(shown(lo), shown(hi)):g}–{max(shown(lo), shown(hi)):g}), so I used the nearest limit.")
            v = min(max(v, lo), hi)
        out[f] = v
    return out, notes


def describe_change(f, v):
    node = {1: "Power Plant", 2: "Residential Area", 3: "Industrial Zone", 4: "Commercial Hub"}[int(f[-1])]
    k = re.sub(r"\d", "", f)
    shown = abs(v) if k == "p" else v
    unit = " s" if k == "tau" else (" p.u." if k == "p" else "")
    return f"{node} {KIND_LABEL[k]} → **{shown:.2f}{unit}**"


# ------------------------------------------------------------------ main entry
def answer(message, ctx):
    """ctx keys: meta, nodes, ranges, last (dict|None), stats, predict(row)->prob,
    recommend(row, prob)->(plan, combined), risk_level(prob)->dict, session (dict-like)."""
    msg = (message or "").strip()
    if not msg:
        return reply("Type a question about the project or your results.", SUGGEST[:4])
    low = msg.lower()
    meta, last = ctx["meta"], ctx["last"]

    # ---------- what-if / prediction from text
    changes, notes = parse_changes(msg, ctx["ranges"])
    if changes:
        return what_if(changes, notes, ctx)

    scores = {}
    for name, pat in INTENTS:
        hits = len(re.findall(pat, low))
        if hits:
            scores[name] = hits
    if not scores:
        return fallback(msg, ctx)
    # explicit "why" and "fix" questions win over topic words that appear inside them
    for prefer in ("leakage", "why", "usage", "pdf", "fix", "last", "greet"):
        if prefer in scores:
            intent = prefer
            break
    else:
        order = [n for n, _ in INTENTS]
        intent = max(scores, key=lambda n: (scores[n], -order.index(n)))
    return HANDLERS[intent](ctx)


def reply(text, chips=None, extra=None):
    out = {"reply": text, "chips": chips or []}
    if extra:
        out.update(extra)
    return out


def need_last(ctx):
    return reply("You haven't run a stability check yet. Open **Stability Check**, enter the readings "
                 "(or use a sample), and run it. Then ask me again.", ["What can you do?", "Explain the risk levels"])


# ------------------------------------------------------------------ handlers
def h_greet(ctx):
    name = ctx.get("user") or "there"
    return reply(f"Hi {name}! 👋 I'm the **GridSense Assistant**. I can explain your results, suggest fixes, "
                 "run what-if predictions, and answer questions about the model and the dataset.", SUGGEST[:4])


def h_thanks(ctx):
    return reply("You're welcome! Ask me anything else about the grid or the model. ⚡", SUGGEST[3:])


def h_bye(ctx):
    return reply("Bye! Keep the lights on. ⚡")


def h_help(ctx):
    return reply("Here's what I can do:\n"
                 "- **Your results:** \"What is my last result?\", \"Why is it unstable?\", \"How do I fix it?\"\n"
                 "- **What-if:** \"What if industrial reaction time is 2?\" or \"set tau3=2 and g2=0.3\"\n"
                 "- **History:** \"How many checks were unstable?\"\n"
                 "- **Model:** \"How accurate is the model?\", \"Compare the models\", \"Which features matter most?\"\n"
                 "- **Concepts:** \"What is tau?\", \"What does Critical mean?\", \"Tell me about the dataset\"",
                 SUGGEST)


def h_last(ctx):
    last = ctx["last"]
    if not last:
        return need_last(ctx)
    r = ctx["risk_level"](last["probability"])
    plan = last["plan"]
    txt = (f"Your latest check is **Report #{last['id']}** for **{last['site']}** ({last['created_at']}).\n"
           f"- Result: **GRID {last['label'].upper()}**\n"
           f"- Chance of stability: **{pct(last['probability'])}**\n"
           f"- Risk level: **{r['level']}**. {r['meaning']}\n")
    if plan:
        txt += f"- The action plan has **{len(plan)} step(s)**. Ask *\"how do I fix it?\"* to see them."
    else:
        txt += "- No action is needed. The grid has a healthy safety margin."
    return reply(txt, ["Why is it unstable?" if last["label"] == "unstable" else "Why is it stable?",
                       "How do I make it stable?", "Download the PDF report"],
                 {"link": {"href": f"/report/{last['id']}.pdf", "label": "📄 Download PDF report"}})


def h_why(ctx):
    last = ctx["last"]
    if not last:
        return need_last(ctx)
    inp, prob = last["inputs"], last["probability"]
    slow = [(f, inp[f]) for f in ("tau1", "tau2", "tau3", "tau4") if inp[f] > 5]
    strong = [(f, inp[f]) for f in ("g1", "g2", "g3", "g4") if inp[f] > 0.6]
    nm = lambda f: {1: "Power Plant", 2: "Residential", 3: "Industrial", 4: "Commercial"}[int(f[-1])]
    if last["label"] == "stable":
        txt = (f"Your grid is **stable ({pct(prob)})** because the locations react quickly and calmly to price changes.\n")
        fast = [f"{nm(f)} {inp[f]:.1f} s" for f in ("tau1", "tau2", "tau3", "tau4") if inp[f] <= 5]
        if fast:
            txt += f"- Fast reaction times: {', '.join(fast)}\n"
        txt += "- In this model, **reaction time (τ)** and **price sensitivity (γ)** decide stability. Power demand barely matters."
        return reply(txt, ["What if industrial reaction time is 9?", "Which features matter most?"])
    txt = f"Your grid is **unstable ({pct(prob)})**. The main reasons the model sees:\n"
    if slow:
        txt += "- **Slow reactions:** " + ", ".join(f"{nm(f)} takes {v:.1f} s" for f, v in slow) + \
               ". When people react late to a price change, demand overshoots and the grid starts to oscillate.\n"
    if strong:
        txt += "- **Over-reaction to price:** " + ", ".join(f"{nm(f)} γ = {v:.2f}" for f, v in strong) + \
               ". Strong reactions make the swings bigger.\n"
    if not slow and not strong:
        txt += "- No single reading is extreme. It's the **combination** of reaction times and price sensitivities that pushes the grid over the edge.\n"
    if last["plan"]:
        first = last["plan"][0]
        txt += f"\nThe biggest single fix is: **{first['title']}** ({first['change']})."
    return reply(txt, ["How do I make it stable?", "What if all reaction times are 1?"])


def h_fix(ctx):
    last = ctx["last"]
    if not last:
        return need_last(ctx)
    plan, comb = last["plan"], last["combined"]
    if not plan:
        return reply(f"Your latest grid is already safe (**{pct(last['probability'])}** stable). No action is needed, "
                     "just keep monitoring. ✅", ["What if industrial reaction time is 9?", "What is my last result?"])
    txt = f"Here's the model's action plan for **{last['site']}** (now {pct(last['probability'])} stable):\n"
    for i, s in enumerate(plan, 1):
        txt += f"- **Step {i}: {s['title']}** ({s['change']}) → {pct(s['prob_after'])}\n  _{s['how']}_\n"
    if comb is not None:
        txt += (f"\nAfter all steps, stability becomes **{pct(comb)}**. "
                + ("The feeder is safe to operate. ✅" if comb >= 0.5 else
                   "That's still not enough, so plan load-shedding or backup supply while the changes are made. ⚠️"))
    return reply(txt, ["Why is it unstable?", "Download the PDF report"],
                 {"link": {"href": f"/report/{last['id']}.pdf", "label": "📄 Download PDF report"}})


def h_stats(ctx):
    s = ctx["stats"]
    if not s["total"] and not s["batches"]:
        return reply("Your history is empty so far. Run a **Stability Check** or a **Batch Upload** first.")
    txt = (f"Your history so far:\n"
           f"- **{s['total']}** single checks: **{s['stable']} stable**, **{s['unstable']} unstable**\n")
    if s["total"]:
        txt += f"- Average chance of stability: **{s['avg'] * 100:.1f}%**\n"
        txt += "- Risk levels: " + ", ".join(f"{k} {v}" for k, v in s["risk"].items() if v) + "\n"
    if s["batches"]:
        txt += f"- **{s['batches']}** batch upload(s) covering **{s['batch_grids']:,}** grids ({s['batch_bad']:,} predicted unstable)"
    return reply(txt, ["What is my last result?", "How accurate is the model?"])


def h_accuracy(ctx):
    m = ctx["meta"]
    b = next(r for r in m["results"] if r["model"] == m["best_model"])
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    return reply(f"The deployed model is the **{m['best_model']}**, tested on **{m['dataset']['test_rows']:,} grids it never saw** during training:\n"
                 f"- Accuracy **{b['accuracy'] * 100:.2f}%**\n- Precision **{b['precision']:.3f}**, Recall **{b['recall']:.3f}**\n"
                 f"- F1 score **{b['f1']:.3f}**, ROC-AUC **{b['roc_auc']:.4f}**\n"
                 f"- Confusion matrix: {tn:,} unstable and {tp:,} stable predicted correctly, with {fp} false alarms and {fn} missed.\n"
                 "So it's wrong only about 2 times in 100.", ["Compare the models", "Which features matter most?"])


def h_models(ctx):
    m = ctx["meta"]
    rows = "\n".join(f"- **{r['model']}**: accuracy {r['accuracy'] * 100:.1f}%, F1 {r['f1']:.3f}"
                     + (" ⭐ best" if r["model"] == m["best_model"] else "") for r in m["results"])
    return reply("Four algorithms were trained on the same data and compared:\n" + rows +
                 "\n\nThe **neural network** wins because the stability boundary is strongly non-linear: reaction times "
                 "and price sensitivities interact. Logistic Regression can only draw a straight-line boundary, "
                 "so it reaches just 81%. It has 3 hidden layers (128-64-32 neurons), ReLU, the Adam optimiser and early stopping.",
                 ["How accurate is the model?", "What is data leakage here?"])


def h_features(ctx):
    imp = ctx["meta"]["importance"]
    lab = {"tau": "reaction time", "g": "price sensitivity", "p": "power demand"}
    nm = {"1": "Power Plant", "2": "Residential", "3": "Industrial", "4": "Commercial"}
    rows = "\n".join(f"- {i + 1}. **{lab[re.sub(r'[0-9]', '', d['feature'])]} ({nm[d['feature'][-1]]})**: {d['importance']:.3f}"
                     for i, d in enumerate(imp[:6]))
    return reply("Feature importance (how much the F1 score drops when a feature is shuffled):\n" + rows +
                 "\n\n**Reaction times** matter most, then **price sensitivities**. Power demand has almost no effect. "
                 "Stability depends on *how fast and how strongly* people react, not on *how much* power they use.",
                 ["What is tau?", "What if all reaction times are 1?"])


def h_tau(ctx):
    return reply("**τ (tau) = reaction time**, in seconds (0.5–10 s). It's how long a location takes to change its "
                 "consumption or production after the electricity price changes.\n"
                 "- **Low τ (1–2 s):** automated smart meters or IoT controllers react almost instantly, which is good for stability.\n"
                 "- **High τ (8–10 s):** a slow or manual response makes demand overshoot, and the grid oscillates.\n"
                 "It's the **most important factor** in the model.", ["What is gamma?", "Which features matter most?"])


def h_power(ctx):
    return reply("**p = power**, in per-unit (p.u.).\n"
                 "- **p1:** power the plant **produces** (positive).\n"
                 "- **p2, p3, p4:** power the residential, industrial and commercial areas **consume** (negative in the dataset, shown as positive demand on the website).\n"
                 "Supply always equals demand, so **p1 = −(p2 + p3 + p4)**. That's why p1 is not used as a model input. "
                 "Power demand has very little effect on stability in this model.", ["What is tau?", "What is gamma?"])


def h_gamma(ctx):
    return reply("**γ (gamma) = price elasticity / price sensitivity** (0.05–1.0). It's how strongly a location changes its "
                 "usage when the price changes.\n- **Low γ:** a calm, small adjustment.\n- **High γ:** a strong reaction, which can over-correct "
                 "and amplify swings.\nIt's the **second most important factor** after reaction time. Operators control it through tariff design "
                 "(time-of-day pricing and incentives).", ["What is tau?", "Which features matter most?"])


def h_risk(ctx):
    return reply("GridSense turns the model's probability into 4 risk levels:\n"
                 "- 🟢 **Safe** (≥ 80%): normal operation, steady power for everyone.\n"
                 "- 🟡 **Watch** (50–80%): stable but near the limit, so monitor closely.\n"
                 "- 🟠 **At Risk** (20–50%): oscillations likely, so take preventive action.\n"
                 "- 🔴 **Critical** (< 20%): relays may trip, with a high chance of power cuts. Act immediately.",
                 ["What is my last result?", "How do I make it stable?"])


def h_dataset(ctx):
    d = ctx["meta"]["dataset"]
    return reply(f"**Dataset:** Smart Grid Stability (augmented), from Kaggle / UCI, based on Decentral Smart Grid Control (DSGC) simulations.\n"
                 f"- **{d['rows']:,} rows**, 14 columns, no missing values\n"
                 f"- **{d['unstable']:,} unstable** and **{d['stable']:,} stable** grids\n"
                 "- Inputs: τ1–τ4 (reaction times), p1–p4 (power), γ1–γ4 (price sensitivity). Target: `stabf`\n"
                 "- It's 10,000 original simulations, each repeated with the 3 consumers in all 6 orders\n"
                 f"- Train/test split: **{d['train_rows']:,} / {d['test_rows']:,}**, split by simulation",
                 ["What is data leakage here?", "Which features matter most?"])


def h_leakage(ctx):
    return reply("Three data traps were handled so the accuracy is honest:\n"
                 "1. **`stab` was removed.** `stabf` is just `stab > 0`, so keeping it would give a fake ~100% accuracy (label leakage).\n"
                 "2. **`p1` was removed.** It always equals −(p2+p3+p4), so it adds no new information.\n"
                 "3. **Split by simulation.** The 60,000 rows are 10,000 grids × 6 orders. A random split would put copies of the same grid "
                 "in both train and test, so GroupShuffleSplit keeps every copy on one side.",
                 ["How accurate is the model?", "Tell me about the dataset"])


def h_project(ctx):
    return reply("**GridSense: Smart Grid Stability Prediction using Machine Learning.**\n"
                 "Smart grids use dynamic pricing to balance supply and demand. If homes, factories and plants react too slowly or too "
                 "strongly to price changes, the grid oscillates and protective relays cut power, causing blackouts and load-shedding.\n"
                 "This system lets an operator enter one set of readings per location. A **neural network (≈98% accurate)** predicts "
                 "stability and the risk level, and the model then builds an **action plan** to prevent power cuts. It also has PDF reports, "
                 "history, batch CSV prediction and this assistant.", ["How accurate is the model?", "What can you do?"])


def h_usage(ctx):
    return reply("How to use GridSense:\n"
                 "- **Stability Check:** enter readings in 5 steps, then run the check. You get the result, risk, action plan and what-if chart.\n"
                 "- **PDF report:** on the result page, or for any check in **History**.\n"
                 "- **Batch Upload:** upload a CSV with columns tau1–tau4, p2–p4, g1–g4 (a sample file is on that page) to check many grids at once.\n"
                 "- **Model Insights:** model comparison, feature importance and the confusion matrix.\n"
                 "- **Assistant (me):** ask about results, or say *\"what if tau3 = 2\"*.", SUGGEST[:4])


def h_pdf(ctx):
    last = ctx["last"]
    if not last:
        return reply("PDF reports are created for every stability check. Run a check first, then use **Download PDF report** "
                     "on the result page or in **History**.")
    return reply(f"Here's the PDF report for your latest check (**Report #{last['id']}**, {last['site']}). It includes the "
                 "college header, the readings, the result, the risk level and the action plan. Older reports are in **History**.",
                 ["What is my last result?", "How do I make it stable?"],
                 {"link": {"href": f"/report/{last['id']}.pdf", "label": "📄 Download PDF report"}})


HANDLERS = {"pdf": h_pdf, "greet": h_greet, "thanks": h_thanks, "bye": h_bye, "help": h_help, "last": h_last, "why": h_why,
            "fix": h_fix, "stats": h_stats, "accuracy": h_accuracy, "models": h_models, "features": h_features,
            "tau": h_tau, "power": h_power, "gamma": h_gamma, "risk": h_risk, "dataset": h_dataset,
            "leakage": h_leakage, "project": h_project, "usage": h_usage}


# ------------------------------------------------------------------ what-if
def what_if(changes, notes, ctx):
    last, ranges, sess = ctx["last"], ctx["ranges"], ctx["session"]
    feats = ctx["meta"]["features"]
    # base = previous chat what-if on the same report, else the latest check, else mid-range values
    if last and sess.get("chat_base_id") == last["id"] and sess.get("chat_inputs"):
        base, src = dict(sess["chat_inputs"]), f"your what-if on Report #{last['id']}"
    elif last:
        base, src = dict(last["inputs"]), f"your latest check (Report #{last['id']}, {last['site']})"
    else:
        base = {f: (ranges[f][0] + ranges[f][1]) / 2 for f in feats}
        src = "a typical mid-range grid (you have no checks yet)"
    before = ctx["predict"](base)
    new = dict(base)
    new.update(changes)
    after = ctx["predict"](new)
    sess["chat_inputs"] = new
    sess["chat_base_id"] = last["id"] if last else None
    r = ctx["risk_level"](after)
    arrow = "📈" if after > before + 1e-6 else ("📉" if after < before - 1e-6 else "➖")
    txt = (f"What-if based on {src}:\n" + "\n".join(f"- {describe_change(f, v)}" for f, v in changes.items()) +
           f"\n\n{arrow} Chance of stability: **{pct(before)} → {pct(after)}**\n"
           f"Result: **GRID {'STABLE' if after >= 0.5 else 'UNSTABLE'}**, risk level **{r['level']}**.")
    if notes:
        txt += "\n\n_Note: " + " ".join(notes) + "_"
    txt += "\n\nYou can keep adding changes (they build on this one), or say *\"reset what-if\"*."
    return reply(txt, ["What if all reaction times are 1?", "How do I make it stable?", "Reset what-if"])


# ------------------------------------------------------------------ fallback (optional LLM)
def fallback(msg, ctx):
    low = msg.lower()
    if "reset" in low:
        ctx["session"].pop("chat_inputs", None)
        ctx["session"].pop("chat_base_id", None)
        return reply("What-if reset. New what-ifs will start from your latest check again.", SUGGEST[3:5])
    if re.search(r"\ball (the )?(reaction|response) times?\b", low):
        nums = re.findall(NUM, low)
        if nums:
            v = float(nums[-1])
            changes, notes = parse_changes(" ".join(f"tau{i}={v}" for i in range(1, 5)), ctx["ranges"])
            return what_if(changes, notes, ctx)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            return reply(ask_llm(msg, ctx, key), SUGGEST[:3])
        except Exception:
            pass
    return reply("I'm not sure about that one yet. I'm built for questions about this project and your results. Try one of these:", SUGGEST)


def ask_llm(msg, ctx, key):
    last = ctx["last"]
    context = {
        "project": "GridSense: smart grid stability prediction (4-node DSGC star grid) with a neural network",
        "model_results": ctx["meta"]["results"], "best_model": ctx["meta"]["best_model"],
        "feature_importance": ctx["meta"]["importance"], "dataset": ctx["meta"]["dataset"],
        "user_stats": ctx["stats"],
        "latest_check": None if not last else {k: last[k] for k in ("site", "probability", "label", "inputs", "plan")},
    }
    body = json.dumps({
        "model": os.environ.get("CHAT_MODEL", "claude-haiku-5-5"), "max_tokens": 400,
        "system": "You are GridSense Assistant inside a student's smart grid stability web app. Answer briefly (under 120 words), "
                  "in simple English, using only this context when talking about results: " + json.dumps(context),
        "messages": [{"role": "user", "content": msg}],
    }).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    return "".join(b.get("text", "") for b in data.get("content", [])) or "Sorry, I couldn't answer that."
