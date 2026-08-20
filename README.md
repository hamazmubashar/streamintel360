<div align="center">

# ⚡ StreamIntel 360

### Enterprise Multi-Model AI Suite for Streaming Analytics

**9 connected ML/AI modules — churn prediction, demand forecasting, hybrid recommendations, computer vision, NLP, and LLM-powered decision intelligence — deployed as one live, interactive platform.**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamintel360-h5l8i2mdn5sfphbwsvcsyg.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**[🚀 Try the Live Demo](https://streamintel360-h5l8i2mdn5sfphbwsvcsyg.streamlit.app/)** · [Architecture](#-architecture) · [Modules](#-the-9-modules) · [Setup](#-setup)

</div>

---

<!--
  📸 ADD A SCREENSHOT OR GIF HERE — this is the single highest-impact addition you can make.
  Record a 15-20 second screen capture of the live dashboard (Home tab → Churn prediction → Poster classifier)
  using something like ScreenToGif or LICEcap, save as demo.gif in a /docs or /assets folder, then:
  ![StreamIntel 360 Demo](docs/demo.gif)
-->

## 📖 Overview

Most ML portfolio projects solve **one** problem — a churn classifier, a recommender, a sentiment model. StreamIntel 360 is different: it's a **connected system of 8 AI modules** built the way a real streaming platform (Netflix, Hulu-style) would actually need them to work together — sharing a common artifact contract, orchestrated behind a single FastAPI backend, and surfaced through one live dashboard.

Every model is trained on real data, evaluated against multiple candidate algorithms (not just one), and the winner is auto-selected by an evidence-based metric — never hardcoded.

## 🎯 Problem → Solution

| Business Problem | Module | Solution |
|---|---|---|
| Subscribers churn with no warning | **01 · Churn Prediction** | Classify at-risk subscribers before they cancel |
| Demand spikes strain infrastructure | **02 · Demand Forecasting** | Forecast next-hour viewership to plan capacity ahead |
| Generic recommendations lose engagement | **03 · Recommendation Engine** | Hybrid, weight-optimized personalized recommendations |
| Manual content tagging doesn't scale | **04 · Computer Vision** | Auto-classify movie genres directly from poster art |
| User sentiment is hard to track at scale | **05 · NLP & Sentiment** | Classify review sentiment automatically |
| Technical outputs aren't executive-friendly | **06 · LLM Intelligence** | Auto-generate evidence-grounded executive briefings |
| Insights are siloed across teams | **07 · RAG Decision Engine** | Ask natural-language questions, get evidence-backed answers |
| "Black box" models erode trust | **08 · Explainable AI** | Transparent, real feature/weight-level explanations |

## 🏗️ Architecture

```mermaid
flowchart LR
    subgraph Kaggle["Trained on Kaggle GPU"]
        A[01 Churn] --> B[02 Forecasting]
        B --> C[03 Recommendations]
        C --> D[04 Computer Vision]
        D --> E[05 NLP Sentiment]
    end
    subgraph Local["Orchestrated Locally"]
        F[06 LLM Intelligence] --> G[07 RAG Engine]
        G --> H[08 Explainable AI]
    end
    E --> F
    H --> I[09 Integration API<br/>FastAPI · 10 endpoints]
    I --> J[Streamlit Frontend<br/>Live Dashboard]

    style Kaggle fill:#1D1B57,stroke:#D97706,color:#fff
    style Local fill:#1D1B57,stroke:#D97706,color:#fff
    style I fill:#D97706,stroke:#1D1B57,color:#fff
    style J fill:#0A0E17,stroke:#D97706,color:#fff
```

Every module reads/writes to a standardized artifact structure (`models/`, `preprocessors/`, `metrics/`, `metadata/`), so the API and frontend can load **any** trained model with zero custom glue code per module.

## 🧠 The 9 Modules

<details>
<summary><b>01 · Subscriber Churn Prediction</b> — Logistic Regression · F1 60.9% · ROC-AUC 83.6%</summary>

5 algorithms compared — Logistic Regression, Decision Tree, Random Forest, XGBoost, and a Neural Network. Logistic Regression won on F1 score, chosen specifically for its direct interpretability, which powers the Explainable AI module downstream.

| Metric | Score |
|---|---|
| Accuracy | 80.4% |
| Precision | 64.8% |
| Recall | 57.5% |
| F1 Score | 60.9% |
| ROC-AUC | 83.6% |

</details>

<details>
<summary><b>02 · Demand Forecasting</b> — LSTM · RMSE 12,779 · MAPE 8.22%</summary>

24-hour sequence forecasting across 5 architectures — SimpleRNN, LSTM, GRU, plus ARIMA/SARIMA statistical baselines — using 17 engineered temporal features (calendar encodings, lag_1/24/168, rolling statistics). LSTM selected by lowest RMSE — deliberately chosen over MAE to penalize rare, high-risk forecasting misses more heavily.

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| **LSTM** | 9,234 | **12,779** | 8.22% |
| GRU | 11,306 | 14,776 | 8.89% |
| SimpleRNN | 11,425 | 15,818 | 8.51% |

</details>

<details>
<summary><b>03 · Hybrid Recommendation Engine</b> — Collaborative + Content + Popularity + SVD</summary>

4 signals blended via grid-search-optimized weights. The optimization genuinely changed the model: collaborative filtering's weight rose from a hardcoded 45% default to a data-driven **70%** optimum — real, provable evidence the tuning mattered.

| Metric | Value |
|---|---|
| MAP@10 | 0.45% |
| Hit Rate@10 | 2.33% |
| Precision@10 | 0.23% |

</details>

<details>
<summary><b>04 · Computer Vision — Poster Genre Classification</b> — EfficientNetB0 · Micro F1 52.8%</summary>

4 CNN architectures benchmarked (Custom CNN, MobileNetV2, EfficientNetB0, ResNet50V2) on 23-class multi-label genre classification, 299×299 input, per-class decision thresholds. EfficientNetB0 won on Micro F1 — notably, ResNet50V2 scored 97.8% recall but only 10.2% precision, a textbook case of why single-metric evaluation is misleading.

</details>

<details>
<summary><b>05 · NLP & Sentiment Analysis</b> — Logistic Regression + TF-IDF · 91.1% Accuracy</summary>

IMDb 50K review dataset, binary sentiment classification. Custom negation-aware preprocessing deliberately preserves words like "not" and "never" that standard stopword removal would strip — preventing "not good" from collapsing into "good."

| Metric | Score |
|---|---|
| Accuracy | 91.1% |
| ROC-AUC | 97.0% |

</details>

<details>
<summary><b>06 · LLM Executive Intelligence</b> — Gemini 3.1 Flash Lite · Inference-only</summary>

Zero-training, evidence-grounded report generation — synthesizes verified metrics from Modules 01-05 into an executive briefing in under 5 seconds, with a strict no-hallucination prompt policy.

</details>

<details>
<summary><b>07 · RAG Decision Engine</b> — TF-IDF Retrieval + Gemini Generation</summary>

Natural-language Q&A grounded strictly in retrieved project evidence — retrieval via TF-IDF + cosine similarity, generation constrained to only use what was retrieved.

</details>

<details>
<summary><b>08 · Explainable AI</b> — Coefficient Analysis + Weight Comparison</summary>

Real, verifiable transparency: per-feature churn contribution scores, and a genuine before/after comparison of recommendation engine weights (hardcoded default vs. grid-search optimum).

</details>

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **ML / DL** | scikit-learn, XGBoost, TensorFlow/Keras |
| **NLP / LLM** | TF-IDF, Google Gemini 3.1 |
| **Backend** | FastAPI |
| **Frontend** | Streamlit + Plotly |
| **Training** | Kaggle GPU |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | Git + Git LFS |

## ⚙️ Engineering Challenges Solved

Real deployment isn't just training models — here's what actually had to be debugged:

- **Cross-environment `pickle` resolution** — the recommendation engine was serialized while running as Python's `__main__`; loading it from an imported module required an explicit `sys.modules["__main__"]` registration workaround.
- **Platform incompatibilities navigated** — evaluated Hugging Face Spaces (requires paid tier for server-side Gradio) and worked through a confirmed Streamlit Cloud platform bug (forced an incompatible Python version) before reaching a stable deployment.
- **Dependency version conflicts** — resolved a `starlette`/`python-multipart` mismatch that silently broke file-upload endpoints.
- **Kaggle → local artifact handoff** — designed a standardized artifact contract so models trained on Kaggle GPU integrate seamlessly with a locally-orchestrated API layer.

## 🚀 Setup

```bash
git clone https://github.com/hamazmubashar/streamintel360.git
cd streamintel360
pip install -r requirements.txt
```

**Run the dashboard locally:**
```bash
streamlit run streamlit_app_v2.py
```

**Run the API standalone:**
```bash
uvicorn app_main:app --reload
# Interactive docs at http://127.0.0.1:8000/docs
```

**Environment variables required:**
```
GEMINI_API_KEY=your_key_here
```

## 📁 Project Structure

```
streamintel360/
├── app_main.py              # FastAPI backend — 10 endpoints, all 9 modules
├── streamlit_app_v2.py      # Streamlit frontend — live dashboard
├── requirements.txt
├── Dockerfile
├── artifacts/                # Trained models, scalers, metrics (Git LFS)
│   ├── notebook_01_churn/
│   ├── notebook_02_forecasting/
│   ├── notebook_03_recommendations/
│   ├── notebook_04_computer_vision/
│   ├── notebook_05_nlp_sentiment/
│   ├── notebook_06_llm/
│   ├── notebook_07_rag/
│   └── notebook_08_explainable_ai/
└── notebooks/                 # Full training pipeline, Kaggle + local
```

## 📊 Results Summary

| Module | Production Model | Key Metric | Result |
|---|---|---|---|
| 01 Churn | Logistic Regression | F1 Score | 60.9% |
| 02 Forecasting | LSTM | RMSE | 12,779 |
| 03 Recommendations | Hybrid (Collab+Content+Pop+SVD) | MAP@10 | 0.45% |
| 04 Computer Vision | EfficientNetB0 (fine-tuned) | Micro F1 | 52.8% |
| 05 NLP Sentiment | Logistic Regression + TF-IDF | Accuracy | 91.1% |
| 06 LLM Intelligence | Gemini 3.1 Flash Lite | Policy | Zero-hallucination |
| 07 RAG | TF-IDF + Gemini | Policy | Evidence-grounded |
| 08 Explainable AI | Coefficient + weight analysis | Coverage | Churn + Recommendations |

## 🔮 Future Improvements

- [ ] Improve churn recall (currently 57.5%) via threshold tuning / cost-sensitive learning
- [ ] Real-time feature pipeline for forecasting (currently uses batch-engineered features)
- [ ] Expand explainability to remaining modules (CV, NLP)
- [ ] CI/CD pipeline for automated retraining

## 👤 Author

**Hamaz Mubashar**
[LinkedIn](https://linkedin.com/in/hamazmubashar) · [GitHub](https://github.com/hamazmubashar) · hamazmubashar2@gmail.com

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

If this project was useful or interesting, consider ⭐ starring the repo!

</div>
