# SmartML Studio

**Intelligent End-to-End Machine Learning Platform for Structured Tabular Data**

SmartML Studio is a Streamlit-based ML platform that guides users through the complete
machine learning workflow — from data upload to model deployment — with intelligent,
confidence-scored recommendations at every step.

## Key Features

- **Smart Advisors**: Every module recommends actions with confidence scores (⭐ + %)
  and plain-English explanations. Nothing auto-executes — the user always decides.
- **Universal Dataset Support**: Works with any structured tabular dataset.
- **Custom Implementations**: From-scratch SVM and kNN implementations alongside
  sklearn/XGBoost/LightGBM/CatBoost.
- **Hybrid Ensemble**: Combine multiple models via majority/weighted voting.
- **Explainable AI**: SHAP-based explanations for every prediction.

## Tech Stack

| Layer          | Technology                                      |
| -------------- | ----------------------------------------------- |
| Frontend       | Streamlit                                       |
| Backend        | Python (OOP, modular)                           |
| ML Libraries   | Scikit-learn, XGBoost, LightGBM, CatBoost       |
| Custom Algos   | SVM (hinge loss + GD), kNN (from scratch)       |
| Explainability | SHAP                                            |
| Visualization  | Plotly, Matplotlib, Seaborn                     |

## Modules

1. 📤 Dataset Upload
2. 🔍 Intelligent Dataset Analysis
3. 📊 Smart Visualization Advisor
4. 🔧 Smart Preprocessing Advisor
5. ⚙️ Feature Engineering
6. 🧠 Smart Model Advisor
7. 🏋️ Model Training
8. 📈 Model Comparison
9. 🎯 Prediction (Single Model / Hybrid Ensemble)
10. 🔬 Explainable AI
11. 💾 Download & Report

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Project Structure

```
smartml-studio/
├── app.py                  # Streamlit entry point
├── config/                 # Settings and logging configuration
├── src/
│   ├── profiling/          # Dataset profiler
│   ├── advisors/           # Recommendation engines
│   ├── models/             # ML models + custom implementations
│   ├── ensemble/           # Hybrid ensemble
│   └── explainability/     # SHAP explanations
├── pages/                  # Streamlit multipage app pages
├── utils/                  # Session state helpers
└── tests/                  # Pytest test suite
```
