"""
STREAMINTEL 360 -- Streamlit Frontend

Interactive demo interface for all 5 trained models plus LLM
executive intelligence, RAG decision engine, and explainable AI.
Deployed on Streamlit Community Cloud.
"""

import os
import io
import json
import re
from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------------------
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

CHURN_DIR = ARTIFACTS_DIR / "notebook_01_churn"
FORECAST_DIR = ARTIFACTS_DIR / "notebook_02_forecasting"
RECOMMEND_DIR = ARTIFACTS_DIR / "notebook_03_recommendations"
VISION_DIR = ARTIFACTS_DIR / "notebook_04_computer_vision"
NLP_DIR = ARTIFACTS_DIR / "notebook_05_nlp_sentiment"
LLM_DIR = ARTIFACTS_DIR / "notebook_06_llm"
RAG_DIR = ARTIFACTS_DIR / "notebook_07_rag"
XAI_DIR = ARTIFACTS_DIR / "notebook_08_explainable_ai"

st.set_page_config(page_title="STREAMINTEL 360", page_icon="📺", layout="wide")

# ------------------------------------------------------------
# CACHED MODEL LOADERS (Streamlit caches these across reruns)
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
    return model, scaler_X, scaler_y, config, best_name


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

    # Register under __main__ so joblib/pickle can resolve the class
    # (it was originally pickled while running as __main__ in the
    # training notebook).
    sys.modules["__main__"].StreamIntelRecommenderEngine = StreamIntelRecommenderEngine

    artifacts = joblib.load(RECOMMEND_DIR / "models" / "recommendation_engine_v1.joblib")
    return artifacts["engine"]


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
    return model, vectorizer


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
# HEADER
# ------------------------------------------------------------
st.title("📺 STREAMINTEL 360")
st.caption("AI-Powered Streaming Analytics & Decision Intelligence Platform")

tabs = st.tabs([
    "Churn Predictor", "Demand Forecast", "Recommendations",
    "Poster Genre Classifier", "Sentiment Analyzer",
    "Executive Report", "Ask the Data (RAG)", "Explainability",
])

# ------------------------------------------------------------
# TAB 1: CHURN PREDICTOR
# ------------------------------------------------------------
with tabs[0]:
    st.header("Subscriber Churn Risk Predictor")
    model, scaler, feature_columns, metadata = load_churn()
    st.caption(f"Model in use: **{metadata.get('best_classical_model', 'Unknown')}**")

    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.number_input("Tenure (months)", 0, 120, 8)
        monthly_fee = st.number_input("Monthly fee (USD)", 0.0, 500.0, 85.0)
    with col2:
        total_spend = st.number_input("Total spend (USD)", 0.0, 20000.0, 680.0)
        internet_fiber = st.checkbox("Has Fiber Optic Internet", value=True)
    with col3:
        streaming_tv = st.checkbox("Uses Streaming TV", value=True)
        support_tickets = st.checkbox("Filed Support Tickets", value=True)

    if st.button("Predict Churn Risk", type="primary"):
        raw_input = {
            "tenure_months": tenure,
            "monthly_fee_usd": monthly_fee,
            "total_spend_usd": total_spend,
            "InternetService_Fiber optic": int(internet_fiber),
            "StreamingTV_Yes": int(streaming_tv),
            "streaming_support_tickets_Yes": int(support_tickets),
        }
        row = {col: raw_input.get(col, 0) for col in feature_columns}
        df = pd.DataFrame([row], columns=feature_columns)

        model_name = metadata.get("best_classical_model", "")
        X = scaler.transform(df) if "Logistic Regression" in model_name else df.values

        probability = float(model.predict_proba(X)[0, 1])
        risk = "🔴 HIGH" if probability >= 0.6 else ("🟡 MEDIUM" if probability >= 0.35 else "🟢 LOW")

        st.metric("Churn Probability", f"{probability:.1%}")
        st.subheader(f"Risk Level: {risk}")

# ------------------------------------------------------------
# TAB 2: DEMAND FORECAST
# ------------------------------------------------------------
with tabs[1]:
    st.header("Streaming Demand Forecast")
    st.markdown(
        "This model predicts the **next hour's** viewership demand from "
        "168 hours (1 week) of recent, already-engineered history. "
        "Upload a CSV with the required columns below, or use the "
        "randomly-generated demo data to see the pipeline run end-to-end "
        "(the predicted number won't be meaningful with random data)."
    )

    fc_model, scaler_X, scaler_y, fc_config, best_name = load_forecast()
    window_size = int(fc_config["window_size_hours"])
    required_features = list(fc_config["input_features"])

    st.caption(f"Model in use: **{best_name}** | Window: {window_size} hours | Features: {len(required_features)}")

    use_demo = st.checkbox("Use random demo data (for testing the pipeline)", value=True)

    if use_demo:
        rng = np.random.default_rng(42)
        demo_df = pd.DataFrame(
            rng.uniform(0, 1, size=(window_size, len(required_features))),
            columns=required_features,
        )
        st.dataframe(demo_df.head(5))
        input_df = demo_df
    else:
        uploaded = st.file_uploader(f"Upload a CSV with exactly {window_size} rows and columns: {required_features}", type="csv")
        input_df = pd.read_csv(uploaded) if uploaded else None

    if st.button("Predict Next Hour Demand", type="primary"):
        if input_df is None or len(input_df) != window_size:
            st.error(f"Need exactly {window_size} rows of history.")
        else:
            X = input_df[required_features].values
            X_scaled = scaler_X.transform(X)
            X_seq = X_scaled.reshape(1, window_size, len(required_features))
            pred_scaled = fc_model.predict(X_seq, verbose=0).flatten()
            pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
            st.metric("Predicted Viewership Demand", f"{pred:,.2f}")

# ------------------------------------------------------------
# TAB 3: RECOMMENDATIONS
# ------------------------------------------------------------
with tabs[2]:
    st.header("Hybrid Movie Recommendation Engine")
    engine = load_recommender()

    rec_mode = st.radio("Recommendation mode", ["Popular picks (no login)", "By user ID", "By seed movie ID"])

    user_id_input = None
    seed_movie_input = None
    if rec_mode == "By user ID":
        user_id_input = st.text_input("User ID", placeholder="e.g. user_01741")
    elif rec_mode == "By seed movie ID":
        seed_movie_input = st.text_input("Seed Movie ID")

    top_n = st.slider("Number of recommendations", 3, 20, 10)

    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Computing recommendations..."):
            results = engine.predict(
                user_id=user_id_input or None,
                seed_movie_id=seed_movie_input or None,
                top_n=top_n,
            )
        st.dataframe(results, use_container_width=True)

# ------------------------------------------------------------
# TAB 4: POSTER GENRE CLASSIFIER
# ------------------------------------------------------------
with tabs[3]:
    st.header("Movie Poster Genre Classifier")
    vision_model, vision_config = load_vision()
    target_size = tuple(vision_config["target_size"])
    classes = vision_config["classes"]
    per_class_thresholds = vision_config.get("per_class_thresholds", {})
    global_threshold = vision_config.get("best_threshold", 0.5)

    st.caption(f"Model: **{vision_config.get('model_name', 'EfficientNetB0')}** | Input size: {target_size[0]}x{target_size[1]}")

    uploaded_image = st.file_uploader("Upload a movie poster", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        st.image(uploaded_image, width=250)
        if st.button("Classify Genres", type="primary"):
            import tensorflow as tf
            image_bytes = uploaded_image.read()
            image = tf.image.decode_jpeg(image_bytes, channels=3) if uploaded_image.type == "image/jpeg" else tf.image.decode_png(image_bytes, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize_with_pad(image, target_size[0], target_size[1])
            image = tf.expand_dims(image, axis=0)

            probabilities = vision_model.predict(image, verbose=0)[0]
            predicted = [
                (genre, float(prob)) for genre, prob in zip(classes, probabilities)
                if prob >= per_class_thresholds.get(genre, global_threshold)
            ]
            predicted.sort(key=lambda x: x[1], reverse=True)

            if predicted:
                st.subheader("Predicted Genres")
                for genre, prob in predicted:
                    st.write(f"**{genre}** — {prob:.1%}")
            else:
                st.info("No genres exceeded the confidence threshold.")

# ------------------------------------------------------------
# TAB 5: SENTIMENT ANALYZER
# ------------------------------------------------------------
with tabs[4]:
    st.header("Review Sentiment Analyzer")
    sentiment_model, sentiment_vectorizer = load_sentiment()

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

    review_text = st.text_area("Paste a movie/show review", height=120, placeholder="This series was absolutely gripping from start to finish...")
    if st.button("Analyze Sentiment", type="primary"):
        cleaned = preprocess_review(review_text)
        vectorized = sentiment_vectorizer.transform([cleaned])
        prediction = sentiment_model.predict(vectorized)[0]
        probability = sentiment_model.predict_proba(vectorized)[0, 1]

        label = "😀 Positive" if prediction == 1 else "😞 Negative"
        confidence = probability if prediction == 1 else 1 - probability
        st.subheader(label)
        st.metric("Confidence", f"{confidence:.1%}")

# ------------------------------------------------------------
# TAB 6: EXECUTIVE REPORT (LLM-generated)
# ------------------------------------------------------------
with tabs[5]:
    st.header("Executive Intelligence Briefing")
    st.caption("Generated by Gemini from verified project metrics — Notebook 06")
    report = load_executive_report()
    st.markdown(report.get("executive_report_markdown", "Report not available."))

# ------------------------------------------------------------
# TAB 7: RAG DECISION ENGINE (live Gemini call)
# ------------------------------------------------------------
with tabs[6]:
    st.header("Ask the Data")
    st.caption("Evidence-grounded Q&A over the project's verified results — Notebook 07")

    gemini_api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)

    if not gemini_api_key:
        st.warning("GEMINI_API_KEY not configured. Add it under Streamlit Cloud's App Settings -> Secrets.")
    else:
        from google import genai
        from sklearn.metrics.pairwise import cosine_similarity

        documents, rag_vectorizer, rag_matrix = load_rag_index()
        client = genai.Client(api_key=gemini_api_key)

        query = st.text_input("Ask a question about the project's results", placeholder="Which forecasting model performed best?")
        if st.button("Ask", type="primary") and query:
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
                    st.caption(d["content"][:300] + "...")

# ------------------------------------------------------------
# TAB 8: EXPLAINABILITY
# ------------------------------------------------------------
with tabs[7]:
    st.header("Explainable AI")
    churn_xai, rec_xai = load_xai_reports()

    st.subheader("Churn Prediction Explanation")
    st.write(f"Sample prediction: **{churn_xai.get('prediction')}** ({churn_xai.get('churn_probability', 0):.1%} probability)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Factors pushing toward churn:**")
        for driver in churn_xai.get("positive_churn_drivers", []):
            st.write(f"• {driver['feature']}: {driver['contribution']:+.4f}")
    with col2:
        st.markdown("**Factors pushing away from churn:**")
        for driver in churn_xai.get("negative_churn_drivers", []):
            st.write(f"• {driver['feature']}: {driver['contribution']:+.4f}")

    st.divider()
    st.subheader("Recommendation Engine Weight Analysis")
    st.write(f"Default weights (pre-optimization): {rec_xai.get('default_weights_pre_optimization')}")
    st.write(f"Optimal weights (grid search): {rec_xai.get('optimal_weights_post_grid_search')}")
    st.info(rec_xai.get("interpretation", {}).get("configuration_note", ""))
