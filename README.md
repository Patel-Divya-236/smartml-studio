<div align="center">

# 🧠 SmartML Studio

**An intelligent, end-to-end machine learning platform for structured tabular data**

Upload a dataset and SmartML Studio walks you through the whole workflow — profiling,
visualisation, preprocessing, feature engineering, model selection, training, ensembling
and explainability — recommending the next action at every step with a **confidence score**
and a **plain-English reason**. Nothing runs automatically; you always make the call.

<br>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

![XGBoost](https://img.shields.io/badge/XGBoost-1B7EC1?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-7CB342?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge&logoColor=black)
![SHAP](https://img.shields.io/badge/SHAP-1F77B4?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

</div>

---

## Table of Contents

- [Why SmartML Studio](#why-smartml-studio)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [The 11-Step Workflow](#the-11-step-workflow)
- [Getting Started](#getting-started)
- [Optional: AI-Powered Planning & Explanations](#optional-ai-powered-planning--explanations)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why SmartML Studio

Most AutoML tools are a black box: you hand over a CSV and get a model back with no
insight into *why* those choices were made. Notebooks are the opposite — total control,
but you rebuild the same boilerplate every single time.

SmartML Studio sits in between. Every module is an **advisor**, not an autopilot:

> *"I recommend **median imputation** for `age` — ⭐⭐⭐⭐ (87%). The column is 12% missing
> and right-skewed (skew = 1.8), so the mean would be pulled toward the tail."*

You see the recommendation, the confidence and the reasoning — then you decide.

---

## Key Features

| | Feature | What it does |
|---|---|---|
| 🎯 | **Confidence-scored advisors** | Every suggestion ships with a ⭐ rating, a percentage and a plain-English justification. |
| 🤖 | **LLM planning layer** | An optional AI planner proposes a full preprocessing, feature-engineering and model plan — every item is deterministically validated before the user sees it. |
| 📂 | **Universal dataset support** | Any structured tabular file (CSV / Excel) — classification or regression, auto-detected. |
| 🔬 | **Deep dataset profiling** | Type inference, missingness, skew, cardinality, outlier and class-imbalance checks. |
| 🛠️ | **From-scratch algorithms** | Vectorised **SVM** (hinge loss + gradient descent) and **kNN**, both written in pure NumPy alongside the library versions. |
| 🌲 | **Gradient boosting suite** | scikit-learn, XGBoost, LightGBM and CatBoost trained and compared side by side. |
| ⚡ | **Real-time training** | WebSocket-powered live progress — watch each model train, with a polling fallback for reliability. |
| 🧠 | **Memory-aware scheduling** | The server estimates each model's memory cost and refuses jobs that would OOM-kill the process, keeping all sessions alive. |
| 🤝 | **Hybrid ensemble** | Combine any trained models via majority or weighted voting. |
| 💡 | **Explainable AI** | SHAP attributions for any individual prediction, plus global feature importance. |
| 💾 | **Session persistence** | Checkpointed sessions survive server restarts — users resume where they left off, not from re-upload. |
| 📄 | **One-click report** | Export the trained model and a full pipeline report of every decision made. |
| 🐳 | **Production-ready deploy** | Dockerised backend on EC2 with Caddy auto-TLS, static frontend on Vercel, DuckDNS for dynamic IPs. |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 19 + TypeScript + Vite + Zustand — single-page workspace, light/dark design system |
| **Backend** | FastAPI — REST + WebSocket for live training progress |
| **Language** | Python 3.10+ — modular and object-oriented |
| **ML libraries** | scikit-learn · XGBoost · LightGBM · CatBoost |
| **Custom algorithms** | SVM (hinge loss + GD) · kNN — implemented from scratch in NumPy |
| **Explainability** | SHAP |
| **Data** | pandas · NumPy · openpyxl |
| **Visualisation** | Recharts · Plotly · Matplotlib · Seaborn |
| **State management** | Zustand (frontend) · session store with disk checkpoints (backend) |
| **Infrastructure** | Docker · Docker Compose · Caddy (auto-TLS) · Vercel · DuckDNS |
| **Runtime safety** | psutil-based memory guard · OMP / OpenBLAS thread pinning |
| **Testing** | pytest — 16 test modules covering every pipeline stage |
| **Optional LLM layer** | Any OpenAI-compatible provider (Groq, OpenRouter, Together, Cerebras, …) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Browser (Vercel)                        │
│   React 19 + TypeScript + Zustand    Recharts · Light/Dark UI   │
└─────────────────────────┬────────────────────────────────────────┘
                          │  REST + WebSocket
┌─────────────────────────▼────────────────────────────────────────┐
│                     FastAPI Backend (EC2 / Docker)               │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ Advisors │  │  Planner  │  │ Trainer  │  │ Explainer      │  │
│  │ (rules)  │  │ (LLM+val) │  │ (thread) │  │ (SHAP)         │  │
│  └──────────┘  └───────────┘  └──────────┘  └────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │  Session Store · Memory Guard · Disk Checkpoints            ││
│  └──────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────-┘
```

---

## The 11-Step Workflow

```
 1. 📤  Dataset Upload            →  CSV / Excel ingestion and validation
 2. 🔍  Dataset Analysis          →  Profiling, quality report, target detection
 3. 📊  Visualisation Advisor     →  Recommends the right plot for each column pair
 4. 🔧  Preprocessing Advisor     →  Imputation, encoding, scaling, outlier strategy
 5. ⚙️  Feature Engineering       →  Interactions, binning, datetime and text features
 6. 🧠  Model Advisor             →  Ranks candidate models for *this* dataset
 7. 🏋️  Model Training            →  Trains the selected models, tracks every run
 8. 📈  Model Comparison          →  Metrics table, ROC / PR curves, confusion matrices
 9. 🎯  Prediction                →  Single model or hybrid ensemble inference
10. 🔬  Explainable AI            →  SHAP waterfall + global importance
11. 💾  Download & Report         →  Serialised model + full pipeline report
```

---

## Getting Started

### Prerequisites

- Python **3.10 or newer**
- `pip`, and a virtual environment (recommended)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Patel-Divya-236/smartml-studio.git
cd smartml-studio

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install frontend dependencies
cd frontend && npm install && cd ..

# 5. Launch both servers
python run_dev.py
```

The app opens at **http://localhost:5173**, the API docs at
**http://127.0.0.1:8000/docs**. Upload any CSV and start at step 1.

To run them separately:

```bash
python -m uvicorn backend.main:app --reload --port 8000
cd frontend && npm run dev
```

---

## Optional: AI-Powered Planning & Explanations

SmartML Studio has two optional AI layers, both powered by any OpenAI-compatible provider:

### 1. LLM Planning Layer

At the analysis step the planner can propose a full preprocessing, feature-engineering and
model plan for the dataset. Every item in the plan passes a **deterministic validator**
(`src/planning/validator.py`) before it reaches the user — invalid techniques, wrong column
types and out-of-range parameters are silently rejected and the rule-based advisors fill
any gaps. The planner proposes; it never decides.

### 2. Plain-English Narration

Several screens can add plain-English narration — why an advisor recommended a step, what
drives a model overall, why one prediction came out as it did, and an executive summary in
the report.

### Setup

**Both layers are entirely optional.** Without a key the app behaves exactly as it
otherwise would, using the rule-based advisors and static explanations.

```bash
cp secrets.toml.example secrets.toml
# then edit the file and add your provider API key
```

Any OpenAI-compatible provider works. Defaults live in `config/settings.py`; the
`LLM_BASE_URL` and `LLM_MODEL` settings override them.

### Guardrails

- The LLM **only rewrites numbers the pipeline already computed** as prose. It never
  sets a confidence score or changes a preprocessing decision — those stay with the
  rule-based advisors.
- The planner's proposals pass a deterministic validator; anything rejected is silently
  replaced by the rule-based ranking.
- **What is sent**: aggregate statistics, the model comparison table, and — only when you
  click *"Generate explanation"* — the SHAP attributions for that one prediction.
- **The uploaded dataset is never transmitted.** This is enforced by a test:
  `tests/test_llm_context.py`.

---

## Deployment

The frontend is a static bundle on **Vercel**. The backend runs as a Dockerised Python
process on a single **EC2** instance behind **Caddy** (auto-TLS).

```bash
cp .env.example .env          # fill in API_DOMAIN, SMARTML_ALLOWED_ORIGINS, etc.
docker compose up -d --build  # starts the API, Caddy, and DuckDNS updater
```

The `duckdns` service in `docker-compose.yml` re-registers the instance's IP on every
boot, so a stopped-and-started EC2 instance (no Elastic IP) resumes automatically.
Session checkpoints live in a Docker volume and survive restarts.

See [DEPLOYMENT.md](DEPLOYMENT.md) for the full operations guide.

---

## Project Structure

```
smartml-studio/
├── run_dev.py                  # Starts the API and the web app together
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── api/                    # Route modules, one per pipeline stage
│   └── core/                   # Session store, JSON serialisation, secrets
├── frontend/
│   └── src/
│       ├── api/                # Typed client
│       ├── store/              # Step order, locking, completion
│       ├── theme/              # Design tokens, light and dark
│       ├── components/         # Shell, primitives, charts
│       └── steps/              # One screen per pipeline module
├── config/
│   ├── settings.py             # Central configuration
│   └── logging_config.py       # Logging setup
├── src/
│   ├── profiling/              # Dataset profiler
│   ├── advisors/               # Recommendation engines (viz, preprocessing, model)
│   ├── planning/               # LLM planner + deterministic validator
│   ├── preprocessing/          # Preprocessing pipeline
│   ├── features/               # Feature engineering
│   ├── models/                 # Model trainer + custom SVM / kNN
│   ├── ensemble/               # Hybrid voting ensemble
│   ├── explainability/         # SHAP explainer
│   ├── evaluation/             # Comparison metrics, feature importance
│   ├── reporting/              # Markdown report builder
│   ├── runtime/                # Memory guard, resource management
│   └── llm/                    # Optional narration layer
├── tests/                      # pytest suite (16 modules)
├── Dockerfile                  # Production container image
├── docker-compose.yml          # API + Caddy + DuckDNS stack
└── Caddyfile                   # Reverse proxy with automatic TLS
```

---

## Testing

```bash
pytest                          # run everything
pytest -v                       # verbose
pytest tests/test_profiler.py  # a single suite
```

The suite covers the custom SVM and kNN implementations, all three advisors, the
profiler, the preprocessing pipeline, feature engineering, the hybrid ensemble, and the
privacy guarantee on the optional LLM layer.

---

## Roadmap

- [x] Docker image and single-host cloud deploy
- [x] LLM-driven planning with deterministic validation
- [x] Memory-aware training scheduler
- [x] Session persistence across restarts
- [ ] Time-series and forecasting support
- [ ] Hyperparameter tuning UI (Optuna)
- [ ] Model registry with experiment versioning
- [ ] Multi-user authentication and role-based access

---

## License

Released under the [MIT License](LICENSE).

<div align="center">
<br>

Built by [**Patel-Divya-236**](https://github.com/Patel-Divya-236)

⭐ Star this repo if you find it useful

</div>
