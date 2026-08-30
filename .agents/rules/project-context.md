# memory.md — SmartML Studio Persistent Project Context

> Place this file at the project root (e.g. `smartml-studio/memory.md`) and additionally copy
> its contents into `smartml-studio/.agents/rules/project-context.md` so Antigravity's Rules
> system loads it automatically in every session (workspace-scoped rules live in
> `<project-root>/.agents/rules/`). Antigravity does not persist ad-hoc chat context between
> sessions — only Rules files are guaranteed to reload — so this file IS the memory.

---

## Role

You are my senior software architect, ML engineer, and UI/UX designer for a Final Year Project.
Follow everything in this file on every task, without needing it repeated in each prompt.

## Project Identity

**SmartML Studio** — Intelligent End-to-End Machine Learning Platform for Structured Tabular Data.

This must NOT look like a generic "upload CSV → train Random Forest" dashboard. It must
demonstrate original software engineering, ML reasoning, and an intelligent recommendation
layer suitable for a final-year viva defense.

Works with ANY structured tabular dataset (diabetes, churn, attrition, house prices, etc.) —
no hardcoded logic for any specific dataset.

## Hard Rules (apply to every task, every module)

1. **Never auto-execute a decision.** Every module recommends; the user accepts, modifies, or
   rejects. Nothing runs without explicit user approval.
2. **Every recommendation needs three things:** a confidence score (⭐ rating + %), a plain-English
   reason grounded in the actual uploaded dataset's properties, and a `Why?` expandable explanation.
3. **No dataset-specific hardcoding.** All logic must derive from dataset profiling at runtime.
4. **Clean architecture, OOP, modular files.** One responsibility per class/module. No duplicated
   logic. Docstrings + logging + graceful error handling everywhere. Production quality, not demo code.
5. **Before writing code for a new module, enter Plan phase first** — produce an
   `implementation_plan.md` describing files to be created/changed, and wait for my approval
   before generating code.

## Tech Stack

- Frontend: React + TypeScript (Vite) over a FastAPI backend
- Backend: Python (OOP, modular)
- ML: Scikit-learn, XGBoost, LightGBM, CatBoost
- Custom algorithms (implemented from scratch, no sklearn internals): SVM, kNN
- Explainability: SHAP
- Visualization: Plotly, Matplotlib, Seaborn

## Core Modules (build in this order)

1. **Dataset Upload** — CSV/Excel upload, target column selection, auto-detect problem type
   (Classification / Regression / Time Series).
2. **Intelligent Dataset Analysis** — rows, columns, missing values, duplicates, dtypes, outliers,
   correlation, cardinality, memory usage, class balance, skewness. Output is reused by every
   later module.
3. **Smart Visualization Advisor** — 4-phase flow: (a) profile dataset, (b) generate a ranked,
   confidence-scored recommendation list with reasons, (c) user accepts/removes/adds/reorders via
   checkboxes, (d) only on "Generate Dashboard" click are charts created.
4. **Smart Preprocessing Advisor** — recommends imputation/encoding/scaling/outlier handling per
   column with confidence score + reason; user can Accept or Change (e.g. swap Median → KNN Imputer).
5. **Feature Engineering** — PCA, feature selection, polynomial features, low-variance removal —
   all user-triggered, none automatic.
6. **Smart Model Advisor** — recommends models based on dataset size, problem type, feature types,
   class balance, complexity; each with confidence score + Why.
7. **Model Training** — sklearn models (RF, Decision Tree, Logistic Regression, Naive Bayes, SVM,
   KNN, XGBoost, LightGBM, CatBoost) + Custom SVM (from scratch) + Custom kNN (from scratch).
8. **Model Comparison** — Accuracy, Precision, Recall, F1, ROC-AUC, training time, prediction time,
   feature importance, side by side.
9. **Prediction** — user picks EITHER a single trained model OR the Custom Hybrid Ensemble
   (majority/weighted voting across selected models). Never forced.
10. **Explainable AI** — SHAP values, feature importance, confusion matrix, ROC curve, learning curve.
11. **Download** — predictions CSV, trained model as `.pkl`, and an evaluation report bundling
    dataset profile + preprocessing applied + models trained + metrics + explainability outputs.

## Original Contributions (do not present as novel: React/FastAPI/sklearn/XGBoost/SHAP)

These ARE the novel parts — treat them as first-class engineering deliverables, not glue code:

- Intelligent Dataset Analyzer
- Visualization Recommendation Engine (confidence-scored, phased, user-gated)
- Smart Preprocessing Advisor (confidence-scored)
- Model Recommendation Advisor (confidence-scored)
- Custom SVM (from scratch)
- Custom kNN (from scratch)
- Custom Hybrid Ensemble (majority/weighted voting)
- The Recommend → Explain Why → User Decides interaction pattern itself

## Coding Standards

- OOP with clear class boundaries (e.g. `DatasetProfiler`, `VisualizationAdvisor`,
  `PreprocessingAdvisor`, `ModelAdvisor`, `HybridEnsemble`, each independently testable).
- Type hints, docstrings, logging via the `logging` module (no bare `print`).
- Config-driven thresholds (e.g. cardinality/skew cutoffs) — not magic numbers buried in logic.
- Errors handled gracefully with user-facing messages, not raw stack traces in the UI.
- Every advisor class exposes a `.recommend(dataset_profile) -> list[Recommendation]` interface,
  where `Recommendation` carries: label, confidence_score, reason, why_explanation.

## What "Done" Looks Like for Any Module

- Code follows the OOP/logging/docstring standards above.
- The module's recommendations are visibly confidence-scored and Why-explained in the UI.
- Nothing auto-runs — there's a visible user action (checkbox, Accept, Generate button) gating
  execution.
- A short section is added to `evaluation_report` covering what that module did, for the final
  Download module to bundle later.
