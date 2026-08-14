"""
STREAMINTEL 360 -- Enterprise Multi-Model Intelligence Suite
Streamlit Frontend (v2 -- upgraded UI)

Built by Hamaz Mubashar
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------------
# PAGE CONFIG & DEVELOPER INFO
# ------------------------------------------------------------
DEVELOPER_NAME = "Hamaz Mubashar"
DEVELOPER_EMAIL = "hamazmubashar2@gmail.com"
DEVELOPER_LINKEDIN = "https://www.linkedin.com/in/PASTE_YOUR_LINKEDIN_HERE"

st.set_page_config(
    page_title="StreamIntel 360 | Enterprise AI Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
CHURN_DIR = ARTIFACTS_DIR / "notebook_01_churn"
FORECAST_DIR = ARTIFACTS_DIR / "notebook_02_forecasting"
RECOMMEND_DIR = ARTIFACTS_DIR / "notebook_03_recommendations"
VISION_DIR = ARTIFACTS_DIR / "notebook_04_computer_vision"
NLP_DIR = ARTIFACTS_DIR / "notebook_05_nlp_sentiment"
LLM_DIR = ARTIFACTS_DIR / "notebook_06_llm"
RAG_DIR = ARTIFACTS_DIR / "notebook_07_rag"
XAI_DIR = ARTIFACTS_DIR / "notebook_08_explainable_ai"

# ------------------------------------------------------------
# CUSTOM DARK THEME CSS
# ------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0a0e17 0%, #0d1220 100%);
        color: #e8e8e8;
    }
    section[data-testid="stSidebar"] {
        background: #0a0e17;
        border-right: 1px solid #1f2937;
    }
    .sidebar-brand {
        font-size: 22px;
        font-weight: 800;
        background: linear-gradient(90deg, #ff512f, #f09819);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .sidebar-subtitle {
        color: #8b93a7;
        font-size: 13px;
        margin-bottom: 20px;
    }
    div[data-testid="stMetric"] {
        background: #131a2b;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="stMetricLabel"] { color: #8b93a7; }
    .stButton > button {
        background: linear-gradient(90deg, #ff4b4b, #ff8c42);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #ff6b6b, #ffa462);
        color: white;
    }
    .footer-credit {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 21rem;
        padding: 12px 20px;
        background: #0a0e17;
        border-top: 1px solid #1f2937;
        font-size: 12px;
        color: #8b93a7;
    }
    .footer-credit a { color: #f09819; text-decoration: none; }
    h1, h2, h3 { color: #f5f5f5; }
    .module-card {
        background: #131a2b;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# CACHED MODEL LOADERS
# ------------------------------------------------------------

@st.cache_resource
def load_churn():
    model = joblib.load(CHURN_DIR / "models" / "best_model.joblib")
    scaler = joblib.load(CHURN_DIR / "preprocessors" / "scaler.joblib")
    features = joblib.load(CHURN_DIR / "preprocessors" / "feature_columns.joblib")
    if isinstance(features, np.ndarray):
        features = features.tolist()
    with open(CHURN_DIR / "metadata" / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, scaler, list(features), metadata


@st.cache_data
def load_churn_comparison():
    with open(CHURN_DIR / "metrics" / "model_comparison.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_forecast():
    from tensorflow import keras
    with open(FORECAST_DIR / "metrics" / "model_comparison.json", "r", encoding="utf-8") as f:
        comparison = json.load(f)
    by_model = {row["Model"]: row for row in comparison}
    dl_models = {"SimpleRNN", "LSTM", "GRU"}
    candidates = {k: v for k, v in by_model.items() if k in dl_models}
    best_name = min(candidates, key=lambda m: candidates[m].get("RMSE", float("inf")))
    export_dir = FORECAST_DIR / "models" / f"{best_name.lower()}_v1"
    model = keras.models.load_model(export_dir / f"{best_name.lower()}_model.keras")
    scaler_X = joblib.load(export_dir / "scaler_X.joblib")
    scaler_y = joblib.load(export_dir / "scaler_y.joblib")
    with open(export_dir / "pipeline_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return model, scaler_X, scaler_y, config, best_name, comparison


@st.cache_resource
def load_recommender():
    import sys
    from sklearn.metrics.pairwise import cosine_similarity

    class StreamIntelRecommenderEngine:
        def __init__(self, interaction_matrix, item_sim_df, content_sim_df,
                     user_profiles, popularity_df, movies_df, tfidf_matrix,
                     weights=(0.50, 0.30, 0.20)):
            self.interaction_matrix = interaction_matrix
            self.item_sim_df = item_sim_df
            self.content_sim_df = content_sim_df
            self.user_profiles = user_profiles
            self.tfidf_matrix = tfidf_matrix
            self.movies_raw = movies_df.copy()
            self.movies_lookup = movies_df.set_index("movie_id") if "movie_id" in movies_df.columns else movies_df
            if "popularity_score" in popularity_df.columns:
                self.popularity_series = popularity_df.set_index("movie_id")["popularity_score"]
            else:
                self.popularity_series = popularity_df
            self.all_movies = (
                self.movies_raw["movie_id"].values if "movie_id" in self.movies_raw.columns
                else self.movies_raw.index.values
            )
            if isinstance(weights, dict):
                self.w_collab = float(weights.get("w_collab", 0.50))
                self.w_content = float(weights.get("w_content", 0.30))
                self.w_pop = float(weights.get("w_pop", 0.20))
            else:
                self.w_collab, self.w_content, self.w_pop = weights

        def _normalize(self, series):
            max_val = series.max()
            return series / max_val if max_val > 0 else series

        def predict(self, user_id=None, seed_movie_id=None, top_n=10):
            is_cold = (user_id is None) or (user_id not in self.interaction_matrix.index)
            if is_cold and seed_movie_id is None:
                top_pop = (
                    self.popularity_series.sort_values(ascending=False)
                    .head(top_n).reset_index()
                    .rename(columns={"index": "movie_id", "popularity_score": "hybrid_score"})
                )
                top_pop["collaborative_score"] = 0.0
                top_pop["content_score"] = 0.0
                top_pop["popularity_score"] = top_pop["hybrid_score"]
                return top_pop.merge(
                    self.movies_raw[["movie_id", "title", "genre_primary", "imdb_rating"]],
                    on="movie_id", how="left",
                )[["movie_id", "title", "genre_primary", "imdb_rating",
                   "collaborative_score", "content_score", "popularity_score", "hybrid_score"]]

            collab_scores = pd.Series(0.0, index=self.all_movies)
            content_scores = pd.Series(0.0, index=self.all_movies)
            pop_scores = self.popularity_series.reindex(self.all_movies, fill_value=0.0)

            if seed_movie_id is not None:
                if seed_movie_id in self.item_sim_df.index:
                    collab_scores = self.item_sim_df.loc[seed_movie_id].reindex(self.all_movies, fill_value=0.0)
                if seed_movie_id in self.content_sim_df.index:
                    content_scores = self.content_sim_df.loc[seed_movie_id].reindex(self.all_movies, fill_value=0.0)
            elif not is_cold:
                user_watched = self.interaction_matrix.loc[user_id][self.interaction_matrix.loc[user_id] > 0]
                valid_items = user_watched.index.intersection(self.item_sim_df.index)
                if len(valid_items) > 0:
                    w = user_watched[valid_items]
                    sim_block = self.item_sim_df.loc[valid_items].reindex(columns=self.all_movies, fill_value=0.0)
                    collab_scores = (sim_block.T @ w) / (w.sum() + 1e-8)
                if user_id in self.user_profiles.index:
                    u_vec = self.user_profiles.loc[user_id].values.reshape(1, -1)
                    sims = cosine_similarity(u_vec, self.tfidf_matrix).flatten()
                    content_scores = pd.Series(sims, index=self.movies_raw["movie_id"]).reindex(self.all_movies, fill_value=0.0)

            norm_collab = self._normalize(collab_scores)
            norm_content = self._normalize(content_scores)
            norm_pop = self._normalize(pop_scores)
            hybrid_scores = self.w_collab * norm_collab + self.w_content * norm_content + self.w_pop * norm_pop

            if not is_cold:
                watched_ids = self.interaction_matrix.loc[user_id][self.interaction_matrix.loc[user_id] > 0].index
                hybrid_scores = hybrid_scores.drop(labels=watched_ids, errors="ignore")
            if seed_movie_id is not None:
                hybrid_scores = hybrid_scores.drop(labels=[seed_movie_id], errors="ignore")

            output = pd.DataFrame({
                "movie_id": hybrid_scores.index.to_numpy(),
                "collaborative_score": norm_collab.reindex(hybrid_scores.index).round(4).to_numpy(),
                "content_score": norm_content.reindex(hybrid_scores.index).round(4).to_numpy(),
                "popularity_score": norm_pop.reindex(hybrid_scores.index).round(4).to_numpy(),
                "hybrid_score": hybrid_scores.round(4).to_numpy(),
            }).reset_index(drop=True)
            output = output.merge(
                self.movies_raw[["movie_id", "title", "genre_primary", "imdb_rating"]],
                on="movie_id", how="left",
            )
            return (
                output.sort_values("hybrid_score", ascending=False).head(top_n)
                .reset_index(drop=True)[["movie_id", "title", "genre_primary", "imdb_rating",
                                          "collaborative_score", "content_score", "popularity_score", "hybrid_score"]]
            )

    sys.modules["__main__"].StreamIntelRecommenderEngine = StreamIntelRecommenderEngine
    artifacts = joblib.load(RECOMMEND_DIR / "models" / "recommendation_engine_v1.joblib")
    return artifacts["engine"], artifacts.get("evaluation_metrics", {})


@st.cache_resource
def load_vision():
    from tensorflow import keras
    model = keras.models.load_model(VISION_DIR / "models" / "efficientnetb0_finetuned_best.keras")
    with open(VISION_DIR / "models" / "efficientnetb0_inference_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return model, config


@st.cache_resource
def load_sentiment():
    import nltk
    for resource in ["stopwords", "wordnet"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    model = joblib.load(NLP_DIR / "models" / "streamintel_sentiment_model.joblib")
    vectorizer = joblib.load(NLP_DIR / "models" / "streamintel_tfidf_vectorizer.joblib")
    with open(NLP_DIR / "models" / "streamintel_sentiment_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, vectorizer, metadata


@st.cache_data
def load_executive_report():
    with open(LLM_DIR / "reports" / "streamintel_executive_report.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_rag_index():
    from sklearn.feature_extraction.text import TfidfVectorizer
    with open(RAG_DIR / "knowledge_base" / "rag_knowledge_base.json", "r", encoding="utf-8") as f:
        documents = json.load(f)
    texts = [f"{d['title']}\n{d['content']}" for d in documents]
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=10000)
    matrix = vectorizer.fit_transform(texts)
    return documents, vectorizer, matrix


@st.cache_data
def load_xai_reports():
    with open(XAI_DIR / "reports" / "churn_explanation.json", "r", encoding="utf-8") as f:
        churn_xai = json.load(f)
    with open(XAI_DIR / "reports" / "recommendation_explanation.json", "r", encoding="utf-8") as f:
        rec_xai = json.load(f)
    return churn_xai, rec_xai

# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">⚡ StreamIntel 360</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Enterprise Multi-Model Intelligence Suite</div>', unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home — Executive Dashboard",
            "📊 01 — Subscriber Churn",
            "📈 02 — Demand Forecasting",
            "🎯 03 & 04 — Recommendation Engine",
            "🎬 05 — Poster Genre Classification",
            "💬 06 — NLP & Sentiment Analysis",
            "🧠 07 — Executive AI Report",
            "🔍 08 — Ask the Data (RAG)",
            "🔬 09 — Explainable AI",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        f'<div class="footer-credit">'
        f'Built by <b>{DEVELOPER_NAME}</b><br>'
        f'<a href="mailto:{DEVELOPER_EMAIL}">{DEVELOPER_EMAIL}</a><br>'
        f'<a href="{DEVELOPER_LINKEDIN}" target="_blank">LinkedIn Profile</a>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# PAGE: HOME — EXECUTIVE DASHBOARD
# ------------------------------------------------------------
if page.startswith("🏠"):
    st.title("Executive Dashboard")
    st.caption("Real-time overview across all 9 STREAMINTEL 360 intelligence modules")

    _, churn_meta = load_churn()[1], load_churn()[3]
    forecast_comparison_data = load_forecast()[5]
    _, rec_metrics = load_recommender()
    _, vision_config = load_vision()
    _, _, nlp_metadata = load_sentiment()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Churn Model", churn_meta.get("best_classical_model", "N/A"))
    with col2:
        best_fc = min(forecast_comparison_data, key=lambda r: r.get("RMSE", float("inf")))
        st.metric("Best Forecast Model", best_fc["Model"], f"RMSE {best_fc.get('RMSE', 0):.2f}")
    with col3:
        st.metric("Recommendation MAP@K", f"{rec_metrics.get('map_at_k', 0):.4f}")
    with col4:
        st.metric("Sentiment Accuracy", f"{nlp_metadata.get('accuracy', 0):.1%}")

    st.divider()

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Forecasting Model Comparison")
        fc_df = pd.DataFrame(forecast_comparison_data)
        fig = px.bar(
            fc_df, x="Model", y="RMSE", color="Model",
            color_discrete_sequence=px.colors.sequential.Oranges_r,
        )
        fig.update_layout(
            plot_bgcolor="#0a0e17", paper_bgcolor="#0a0e17",
            font_color="#e8e8e8", showlegend=False, height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.subheader("NLP Model Performance")
        nlp_metrics_dict = {
            "Accuracy": nlp_metadata.get("accuracy", 0),
            "Precision": nlp_metadata.get("precision", 0),
            "Recall": nlp_metadata.get("recall", 0),
            "F1 Score": nlp_metadata.get("f1_score", 0),
            "ROC-AUC": nlp_metadata.get("roc_auc", 0),
        }
        fig2 = go.Figure(go.Scatterpolar(
            r=list(nlp_metrics_dict.values()),
            theta=list(nlp_metrics_dict.keys()),
            fill="toself",
            line_color="#f09819",
        ))
        fig2.update_layout(
            polar=dict(bgcolor="#131a2b", radialaxis=dict(visible=True, range=[0, 1], color="#8b93a7")),
            paper_bgcolor="#0a0e17", font_color="#e8e8e8", height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Module Status")
    modules = [
        ("01 — Subscriber Churn", "✅ Live", churn_meta.get("best_classical_model", "")),
        ("02 — Demand Forecasting", "✅ Live", best_fc["Model"]),
        ("03/04 — Recommendation Engine", "✅ Live", "Hybrid (Collab + Content + Popularity)"),
        ("05 — Poster Genre Classification", "✅ Live", vision_config.get("model_name", "EfficientNetB0")),
        ("06 — NLP & Sentiment Analysis", "✅ Live", nlp_metadata.get("model", "Logistic Regression")),
        ("07 — Executive AI Report", "✅ Live", "Gemini 3.1 Flash Lite"),
        ("08 — RAG Decision Engine", "✅ Live", "TF-IDF + Gemini"),
        ("09 — Explainable AI", "✅ Live", "Coefficient + Weight Analysis"),
    ]
    status_df = pd.DataFrame(modules, columns=["Module", "Status", "Details"])
    st.dataframe(status_df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# PAGE: 01 — SUBSCRIBER CHURN
# ------------------------------------------------------------
elif page.startswith("📊"):
    st.title("Subscriber Churn Risk Predictor")
    model, scaler, feature_columns, metadata = load_churn()
    model_name = metadata.get("best_classical_model", "")
    st.caption(f"Model in use: **{model_name}**")

    left, right = st.columns([1, 1.2])

    with left:
        tenure = st.slider("Tenure (months)", 0, 72, 8)
        monthly_fee = st.number_input("Monthly charge (USD)", 0.0, 500.0, 70.0, step=5.0)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
        support_tickets = st.selectbox("Filed Support Tickets", ["Yes", "No"])
        predict_clicked = st.button("⚡ Execute Live Churn Prediction", type="primary", use_container_width=True)

    with right:
        if predict_clicked:
            raw_input = {
                "tenure_months": tenure,
                "monthly_fee_usd": monthly_fee,
                "total_spend_usd": monthly_fee * max(tenure, 1),
                "InternetService_Fiber optic": int(internet == "Fiber optic"),
                "InternetService_No": int(internet == "No"),
                "StreamingTV_Yes": int(streaming_tv == "Yes"),
                "StreamingMovies_Yes": int(streaming_movies == "Yes"),
                "streaming_support_tickets_Yes": int(support_tickets == "Yes"),
                "subscription_tier_One year": int(contract == "One year"),
                "subscription_tier_Two year": int(contract == "Two year"),
            }
            row = {col: raw_input.get(col, 0) for col in feature_columns}
            df = pd.DataFrame([row], columns=feature_columns)
            X = scaler.transform(df) if "Logistic Regression" in model_name else df.values
            probability = float(model.predict_proba(X)[0, 1]) * 100

            if probability >= 60:
                risk_label, bar_color = "HIGH RISK", "#ff4b4b"
            elif probability >= 35:
                risk_label, bar_color = "MEDIUM RISK", "#f09819"
            else:
                risk_label, bar_color = "LOW RISK", "#2ecc71"

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability,
                number={"suffix": "%", "font": {"color": bar_color, "size": 48}},
                title={"text": risk_label, "font": {"color": bar_color, "size": 20}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b93a7"},
                    "bar": {"color": bar_color},
                    "bgcolor": "#131a2b",
                    "steps": [
                        {"range": [0, 35], "color": "#1a2332"},
                        {"range": [35, 60], "color": "#241f1a"},
                        {"range": [60, 100], "color": "#2a1616"},
                    ],
                },
            ))
            fig.update_layout(paper_bgcolor="#0a0e17", font_color="#e8e8e8", height=350)
            st.plotly_chart(fig, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Prediction", "Likely to churn" if probability >= 50 else "Likely to stay")
            m2.metric("Churn Probability", f"{probability:.1f}%")
            m3.metric("Risk Level", risk_label)
        else:
            st.info("Fill in the subscriber profile and click **Execute Live Churn Prediction**.")

    st.divider()
    st.subheader("Model Comparison (Training Results)")
    comparison_data = load_churn_comparison()
    comp_df = pd.DataFrame(comparison_data)
    if not comp_df.empty and "F1 Score" in comp_df.columns:
        fig3 = px.bar(comp_df, x="Model", y="F1 Score", color="Model", color_discrete_sequence=px.colors.sequential.Oranges_r)
        fig3.update_layout(plot_bgcolor="#0a0e17", paper_bgcolor="#0a0e17", font_color="#e8e8e8", showlegend=False, height=300)
        st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------
# PAGE: 02 — DEMAND FORECASTING
# ------------------------------------------------------------
elif page.startswith("📈"):
    st.title("Streaming Demand Forecast")
    fc_model, scaler_X, scaler_y, fc_config, best_name, comparison_data = load_forecast()
    window_size = int(fc_config["window_size_hours"])
    required_features = list(fc_config["input_features"])

    st.caption(f"Model: **{best_name}** | Window: {window_size}h | Features: {len(required_features)}")

    st.markdown(
        "This model predicts the **next hour's** demand from 168 hours of engineered history. "
        "Use the random demo data below to verify the pipeline runs end-to-end."
    )

    if st.button("⚡ Run Forecast Pipeline", type="primary"):
        rng = np.random.default_rng(42)
        demo_window = pd.DataFrame(
            rng.uniform(0, 1, size=(window_size, len(required_features))),
            columns=required_features,
        )
        X = demo_window[required_features].values
        X_scaled = scaler_X.transform(X)
        X_seq = X_scaled.reshape(1, window_size, len(required_features))
        pred_scaled = fc_model.predict(X_seq, verbose=0).flatten()
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]

        st.metric("Predicted Viewership Demand (Next Hour)", f"{pred:,.2f}")
        st.caption("Note: uses synthetic demo data, so this number is illustrative of the pipeline, not a real forecast.")

    st.divider()
    st.subheader("Model Comparison (Training Results)")
    comp_df = pd.DataFrame(comparison_data)
    fig = px.bar(comp_df, x="Model", y=["MAE", "RMSE"], barmode="group", color_discrete_sequence=["#f09819", "#ff4b4b"])
    fig.update_layout(plot_bgcolor="#0a0e17", paper_bgcolor="#0a0e17", font_color="#e8e8e8", height=350)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# PAGE: 03 & 04 — RECOMMENDATION ENGINE
# ------------------------------------------------------------
elif page.startswith("🎯"):
    st.title("Hybrid Movie Recommendation Engine")
    engine, rec_metrics = load_recommender()

    mode = st.radio("Mode", ["Popular picks", "By User ID", "By Seed Movie ID"], horizontal=True)
    user_id_in, seed_in = None, None
    if mode == "By User ID":
        user_id_in = st.text_input("User ID", placeholder="e.g. user_01741")
    elif mode == "By Seed Movie ID":
        seed_in = st.text_input("Seed Movie ID")

    top_n = st.slider("Number of recommendations", 3, 20, 10)

    if st.button("⚡ Get Recommendations", type="primary"):
        with st.spinner("Computing recommendations..."):
            results = engine.predict(user_id=user_id_in or None, seed_movie_id=seed_in or None, top_n=top_n)
        st.dataframe(results, use_container_width=True, hide_index=True)

        if not results.empty:
            fig = px.bar(
                results.head(10), x="hybrid_score", y="title", orientation="h",
                color="hybrid_score", color_continuous_scale="Oranges",
            )
            fig.update_layout(plot_bgcolor="#0a0e17", paper_bgcolor="#0a0e17", font_color="#e8e8e8", height=400, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Evaluation Metrics")
    m1, m2 = st.columns(2)
    m1.metric("MAP@K", f"{rec_metrics.get('map_at_k', 0):.4f}")
    m2.metric("Catalog Version", rec_metrics.get("k", "N/A"))

# ------------------------------------------------------------
# PAGE: 05 — POSTER GENRE CLASSIFICATION
# ------------------------------------------------------------
elif page.startswith("🎬"):
    st.title("Movie Poster Genre Classifier")
    vision_model, vision_config = load_vision()
    target_size = tuple(vision_config["target_size"])
    classes = vision_config["classes"]
    per_class_thresholds = vision_config.get("per_class_thresholds", {})
    global_threshold = vision_config.get("best_threshold", 0.5)

    st.caption(f"Model: **{vision_config.get('model_name', 'EfficientNetB0')}** | Input: {target_size[0]}x{target_size[1]}")

    uploaded_image = st.file_uploader("Upload a movie poster", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.image(uploaded_image, use_container_width=True)
        with col2:
            if st.button("⚡ Classify Genres", type="primary"):
                import tensorflow as tf
                image_bytes = uploaded_image.read()
                decode_fn = tf.image.decode_jpeg if uploaded_image.type == "image/jpeg" else tf.image.decode_png
                image = decode_fn(image_bytes, channels=3)
                image = tf.image.convert_image_dtype(image, tf.float32)
                image = tf.image.resize_with_pad(image, target_size[0], target_size[1])
                image = tf.expand_dims(image, axis=0)

                probabilities = vision_model.predict(image, verbose=0)[0]
                results_df = pd.DataFrame({"Genre": classes, "Probability": probabilities}).sort_values("Probability", ascending=False)

                fig = px.bar(
                    results_df.head(10), x="Probability", y="Genre", orientation="h",
                    color="Probability", color_continuous_scale="Oranges",
                )
                fig.update_layout(plot_bgcolor="#0a0e17", paper_bgcolor="#0a0e17", font_color="#e8e8e8", height=400, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

                predicted = results_df[results_df["Probability"] >= results_df["Genre"].map(lambda g: per_class_thresholds.get(g, global_threshold))]
                if not predicted.empty:
                    st.success(f"Predicted genres: {', '.join(predicted['Genre'].tolist())}")
                else:
                    st.info("No genres exceeded the confidence threshold.")

# ------------------------------------------------------------
# PAGE: 06 — NLP & SENTIMENT ANALYSIS
# ------------------------------------------------------------
elif page.startswith("💬"):
    st.title("Review Sentiment Analyzer")
    sentiment_model, sentiment_vectorizer, nlp_metadata = load_sentiment()

    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    negation_words = {"no", "nor", "not", "never", "neither", "none", "nothing", "nowhere", "hardly", "scarcely", "barely"}
    sentiment_stop_words = stop_words - negation_words

    def preprocess_review(text):
        text = re.sub(r"<[^>]+>", " ", text)
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = [
            lemmatizer.lemmatize(w) for w in text.split()
            if w not in sentiment_stop_words and (w in negation_words or len(w) > 2)
        ]
        return " ".join(tokens)

    review_text = st.text_area("Paste a review", height=120, placeholder="This series was absolutely gripping from start to finish...")
    if st.button("⚡ Analyze Sentiment", type="primary"):
        cleaned = preprocess_review(review_text)
        vectorized = sentiment_vectorizer.transform([cleaned])
        prediction = sentiment_model.predict(vectorized)[0]
        probability = sentiment_model.predict_proba(vectorized)[0, 1]
        confidence = probability if prediction == 1 else 1 - probability

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Positive 😀" if prediction == 1 else "Negative 😞"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2ecc71" if prediction == 1 else "#ff4b4b"}, "bgcolor": "#131a2b"},
        ))
        fig.update_layout(paper_bgcolor="#0a0e17", font_color="#e8e8e8", height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{nlp_metadata.get('accuracy', 0):.1%}")
    m2.metric("Precision", f"{nlp_metadata.get('precision', 0):.1%}")
    m3.metric("Recall", f"{nlp_metadata.get('recall', 0):.1%}")
    m4.metric("F1 Score", f"{nlp_metadata.get('f1_score', 0):.1%}")

# ------------------------------------------------------------
# PAGE: 07 — EXECUTIVE AI REPORT
# ------------------------------------------------------------
elif page.startswith("🧠"):
    st.title("Executive Intelligence Briefing")
    st.caption("Generated by Gemini from verified project metrics")
    report = load_executive_report()
    st.markdown(report.get("executive_report_markdown", "Report not available."))

# ------------------------------------------------------------
# PAGE: 08 — ASK THE DATA (RAG)
# ------------------------------------------------------------
elif page.startswith("🔍"):
    st.title("Ask the Data")
    st.caption("Evidence-grounded Q&A over the project's verified results")

    gemini_api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)

    if not gemini_api_key:
        st.warning("GEMINI_API_KEY not configured. Add it under App Settings -> Secrets.")
    else:
        from google import genai
        from sklearn.metrics.pairwise import cosine_similarity

        documents, rag_vectorizer, rag_matrix = load_rag_index()
        client = genai.Client(api_key=gemini_api_key)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(message)

        query = st.chat_input("Ask a question about the project's results...")
        if query:
            st.session_state.chat_history.append(("user", query))
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving evidence and generating answer..."):
                    query_vec = rag_vectorizer.transform([query])
                    sims = cosine_similarity(query_vec, rag_matrix).flatten()
                    top_k = sims.argsort()[::-1][:3]
                    retrieved = [documents[i] for i in top_k]

                    evidence_blocks = "\n".join(
                        f"\n[EVIDENCE {i+1}]\nTitle: {d['title']}\nSource: {d['source']}\n\n{d['content']}\n"
                        for i, d in enumerate(retrieved)
                    )
                    prompt = f"""You are the Decision Intelligence Assistant for STREAMINTEL 360.
Answer using ONLY the retrieved evidence below. If insufficient, say so explicitly.
Do not invent metrics. Distinguish facts from recommendations.

USER QUESTION:
{query}

RETRIEVED EVIDENCE:
{evidence_blocks}

Return a clear, concise, decision-oriented answer."""

                    response = client.models.generate_content(model="gemini-3.1-flash-lite", contents=prompt)
                    answer = getattr(response, "text", None) or "No response generated."

                st.markdown(answer)
                with st.expander("Retrieved evidence"):
                    for d in retrieved:
                        st.markdown(f"**{d['title']}** (source: {d['source']})")

            st.session_state.chat_history.append(("assistant", answer))

# ------------------------------------------------------------
# PAGE: 09 — EXPLAINABLE AI
# ------------------------------------------------------------
elif page.startswith("🔬"):
    st.title("Explainable AI")
    churn_xai, rec_xai = load_xai_reports()

    st.subheader("Churn Prediction Explanation")
    st.write(f"Sample prediction: **{churn_xai.get('prediction')}** ({churn_xai.get('churn_probability', 0):.1%} probability)")

    positive_df = pd.DataFrame(churn_xai.get("positive_churn_drivers", []))
    negative_df = pd.DataFrame(churn_xai.get("negative_churn_drivers", []))

    if not positive_df.empty or not negative_df.empty:
        combined = pd.concat([positive_df, negative_df])
        fig = px.bar(
            combined, x="contribution", y="feature", orientation="h",
            color="contribution", color_continuous_scale=["#2ecc71", "#131a2b", "#ff4b4b"],
        )
        fig.update_layout(plot_bgcolor="#0a0e17", paper_bgcolor="#0a0e17", font_color="#e8e8e8", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Recommendation Engine Weight Analysis")
    default_w = rec_xai.get("default_weights_pre_optimization", {})
    optimal_w = rec_xai.get("optimal_weights_post_grid_search", {})

    weight_df = pd.DataFrame({
        "Component": ["Collaborative", "Content", "Popularity"],
        "Default": [default_w.get("w_collab", 0), default_w.get("w_content", 0), default_w.get("w_pop", 0)],
        "Optimal": [optimal_w.get("w_collab", 0), optimal_w.get("w_content", 0), optimal_w.get("w_pop", 0)],
    })
    fig2 = px.bar(weight_df, x="Component", y=["Default", "Optimal"], barmode="group", color_discrete_sequence=["#8b93a7", "#f09819"])
    fig2.update_layout(plot_bgcolor="#0a0e17", paper_bgcolor="#0a0e17", font_color="#e8e8e8", height=350)
    st.plotly_chart(fig2, use_container_width=True)

    st.info(rec_xai.get("interpretation", {}).get("configuration_note", ""))
