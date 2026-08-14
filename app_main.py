
"""
STREAMINTEL 360 -- Integration API

Wires together all 5 trained models (churn, demand forecasting,
hybrid recommendations, computer vision, NLP sentiment) plus the
LLM executive report, RAG decision engine, and explainable AI
layers into a single FastAPI application.

Run locally with:
    uvicorn app_main:app --reload

Artifacts directory can be overridden with the STREAMINTEL_ARTIFACTS_DIR
environment variable; defaults to the local project path.
"""

import os
import io
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

# ------------------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------------------
ARTIFACTS_DIR = Path(os.getenv("STREAMINTEL_ARTIFACTS_DIR", r"E:\STREAMINTEL360_Complete\artifacts"))

CHURN_DIR = ARTIFACTS_DIR / "notebook_01_churn"
FORECAST_DIR = ARTIFACTS_DIR / "notebook_02_forecasting"
RECOMMEND_DIR = ARTIFACTS_DIR / "notebook_03_recommendations"
VISION_DIR = ARTIFACTS_DIR / "notebook_04_computer_vision"
NLP_DIR = ARTIFACTS_DIR / "notebook_05_nlp_sentiment"
LLM_DIR = ARTIFACTS_DIR / "notebook_06_llm"
RAG_DIR = ARTIFACTS_DIR / "notebook_07_rag"
XAI_DIR = ARTIFACTS_DIR / "notebook_08_explainable_ai"

# ------------------------------------------------------------
# MODULE 1: SUBSCRIBER CHURN
# ------------------------------------------------------------
churn_model = joblib.load(CHURN_DIR / "models" / "best_model.joblib")
churn_scaler = joblib.load(CHURN_DIR / "preprocessors" / "scaler.joblib")
churn_features = joblib.load(CHURN_DIR / "preprocessors" / "feature_columns.joblib")
if isinstance(churn_features, np.ndarray):
    churn_features = churn_features.tolist()
churn_features = list(churn_features)

with open(CHURN_DIR / "metadata" / "metadata.json", "r", encoding="utf-8") as f:
    churn_metadata = json.load(f)
churn_model_name = churn_metadata.get("best_classical_model", "")


def predict_churn(feature_dict: Dict[str, float]) -> Dict[str, Any]:
    row = {col: feature_dict.get(col, 0) for col in churn_features}
    df = pd.DataFrame([row], columns=churn_features)

    # Matches Notebook 01's conditional: only Logistic Regression gets
    # scaled input, tree-based models receive raw feature values.
    if "Logistic Regression" in churn_model_name:
        X = churn_scaler.transform(df)
    else:
        X = df.values

    probability = float(churn_model.predict_proba(X)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "churn_probability": round(probability, 4),
        "prediction": prediction,
        "risk_level": "HIGH" if probability >= 0.6 else ("MEDIUM" if probability >= 0.35 else "LOW"),
        "model_used": churn_model_name,
    }


# ------------------------------------------------------------
# MODULE 2: DEMAND FORECASTING
# ------------------------------------------------------------
from tensorflow import keras  # noqa: E402

with open(FORECAST_DIR / "metrics" / "model_comparison.json", "r", encoding="utf-8") as f:
    forecast_comparison = json.load(f)
_forecast_by_model = {row["Model"]: row for row in forecast_comparison}
_dl_models = {"SimpleRNN", "LSTM", "GRU"}
_dl_candidates = {k: v for k, v in _forecast_by_model.items() if k in _dl_models}
best_forecast_model_name = min(_dl_candidates, key=lambda m: _dl_candidates[m].get("RMSE", float("inf")))

_forecast_export_dir = FORECAST_DIR / "models" / f"{best_forecast_model_name.lower()}_v1"
forecast_model = keras.models.load_model(_forecast_export_dir / f"{best_forecast_model_name.lower()}_model.keras")
forecast_scaler_X = joblib.load(_forecast_export_dir / "scaler_X.joblib")
forecast_scaler_y = joblib.load(_forecast_export_dir / "scaler_y.joblib")
with open(_forecast_export_dir / "pipeline_config.json", "r", encoding="utf-8") as f:
    forecast_config = json.load(f)

FORECAST_WINDOW = int(forecast_config["window_size_hours"])
FORECAST_FEATURES = list(forecast_config["input_features"])


def predict_demand(recent_hours: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    recent_hours: a list of FORECAST_WINDOW dicts, each containing every
    feature in FORECAST_FEATURES for one hour of already-engineered history
    (calendar features, lags, rolling stats -- as produced by Notebook 02's
    feature engineering step). This endpoint predicts the NEXT hour only.
    """
    if len(recent_hours) != FORECAST_WINDOW:
        raise ValueError(
            f"Expected exactly {FORECAST_WINDOW} hourly rows of history, got {len(recent_hours)}."
        )

    df = pd.DataFrame(recent_hours)
    missing_cols = [c for c in FORECAST_FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required features: {missing_cols}")

    X = df[FORECAST_FEATURES].values
    X_scaled = forecast_scaler_X.transform(X)
    X_seq = X_scaled.reshape(1, FORECAST_WINDOW, len(FORECAST_FEATURES))

    pred_scaled = forecast_model.predict(X_seq, verbose=0).flatten()
    pred = forecast_scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]

    return {
        "predicted_viewership_demand": round(float(pred), 2),
        "model_used": best_forecast_model_name,
        "window_size_hours": FORECAST_WINDOW,
    }

# ------------------------------------------------------------
# MODULE 3: HYBRID RECOMMENDATION ENGINE
# ------------------------------------------------------------
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402


class StreamIntelRecommenderEngine:
    """Must match the class definition used when Notebook 03 pickled
    this engine, or joblib.load() cannot resolve the object."""

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
        elif isinstance(weights, (tuple, list)) and len(weights) == 3:
            self.w_collab, self.w_content, self.w_pop = weights
        else:
            raise ValueError("weights must be a dict or 3-element tuple/list")

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


# NOTE: this class was originally pickled while running as __main__ in
# the training notebook, so joblib/pickle expects to find it there --
# not under this module's own name. Registering it under __main__
# explicitly lets joblib.load() resolve it regardless of how this file
# is imported (as a script, via %%run, or import app_main).
import sys as _sys
_sys.modules["__main__"].StreamIntelRecommenderEngine = StreamIntelRecommenderEngine

_recommendation_artifacts = joblib.load(RECOMMEND_DIR / "models" / "recommendation_engine_v1.joblib")
recommendation_engine = _recommendation_artifacts["engine"]


def get_recommendations(user_id: Optional[str], seed_movie_id: Optional[str], top_n: int) -> List[Dict[str, Any]]:
    result_df = recommendation_engine.predict(user_id=user_id, seed_movie_id=seed_movie_id, top_n=top_n)
    return result_df.to_dict(orient="records")

# ------------------------------------------------------------
# MODULE 4: COMPUTER VISION (POSTER GENRE CLASSIFICATION)
# ------------------------------------------------------------
import tensorflow as tf  # noqa: E402

vision_model = keras.models.load_model(VISION_DIR / "models" / "efficientnetb0_finetuned_best.keras")
with open(VISION_DIR / "models" / "efficientnetb0_inference_config.json", "r", encoding="utf-8") as f:
    vision_config = json.load(f)

VISION_TARGET_SIZE = tuple(vision_config["target_size"])
VISION_CLASSES = vision_config["classes"]
VISION_PER_CLASS_THRESHOLDS = vision_config.get("per_class_thresholds", {})
VISION_GLOBAL_THRESHOLD = vision_config.get("best_threshold", 0.5)


def predict_genres(image_bytes: bytes) -> Dict[str, Any]:
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, VISION_TARGET_SIZE[0], VISION_TARGET_SIZE[1])
    image = tf.expand_dims(image, axis=0)

    probabilities = vision_model.predict(image, verbose=0)[0]

    predicted_genres = []
    for genre, prob in zip(VISION_CLASSES, probabilities):
        threshold = VISION_PER_CLASS_THRESHOLDS.get(genre, VISION_GLOBAL_THRESHOLD)
        if prob >= threshold:
            predicted_genres.append({"genre": genre, "probability": round(float(prob), 4)})

    predicted_genres.sort(key=lambda x: x["probability"], reverse=True)
    return {
        "predicted_genres": predicted_genres,
        "all_probabilities": {g: round(float(p), 4) for g, p in zip(VISION_CLASSES, probabilities)},
    }


# ------------------------------------------------------------
# MODULE 5: NLP SENTIMENT
# ------------------------------------------------------------
import nltk  # noqa: E402
from nltk.corpus import stopwords  # noqa: E402
from nltk.stem import WordNetLemmatizer  # noqa: E402

for resource in ["stopwords", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

sentiment_model = joblib.load(NLP_DIR / "models" / "streamintel_sentiment_model.joblib")
sentiment_vectorizer = joblib.load(NLP_DIR / "models" / "streamintel_tfidf_vectorizer.joblib")

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))
_negation_words = {"no", "nor", "not", "never", "neither", "none", "nothing", "nowhere", "hardly", "scarcely", "barely"}
_sentiment_stop_words = _stop_words - _negation_words


def _preprocess_review(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [
        _lemmatizer.lemmatize(w) for w in text.split()
        if w not in _sentiment_stop_words and (w in _negation_words or len(w) > 2)
    ]
    return " ".join(tokens)


def predict_sentiment(reviews: List[str]) -> List[Dict[str, Any]]:
    cleaned = [_preprocess_review(r) for r in reviews]
    vectorized = sentiment_vectorizer.transform(cleaned)
    predictions = sentiment_model.predict(vectorized)
    probabilities = sentiment_model.predict_proba(vectorized)[:, 1]

    results = []
    for raw, pred, prob in zip(reviews, predictions, probabilities):
        results.append({
            "review": raw,
            "sentiment": "positive" if pred == 1 else "negative",
            "confidence": round(float(prob if pred == 1 else 1 - prob), 4),
        })
    return results

# ------------------------------------------------------------
# MODULE 6: LLM EXECUTIVE INTELLIGENCE (static, pre-generated)
# ------------------------------------------------------------
with open(LLM_DIR / "reports" / "streamintel_executive_report.json", "r", encoding="utf-8") as f:
    executive_report = json.load(f)


# ------------------------------------------------------------
# MODULE 7: RAG DECISION ENGINE (live retrieval + live Gemini call)
# ------------------------------------------------------------
from dotenv import load_dotenv  # noqa: E402
from google import genai  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

load_dotenv(ARTIFACTS_DIR.parent / ".env")
_gemini_api_key = os.getenv("GEMINI_API_KEY")
_gemini_client = genai.Client(api_key=_gemini_api_key) if _gemini_api_key else None
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"

with open(RAG_DIR / "knowledge_base" / "rag_knowledge_base.json", "r", encoding="utf-8") as f:
    rag_knowledge_documents = json.load(f)

_rag_documents_text = [f"{d['title']}\n{d['content']}" for d in rag_knowledge_documents]
_rag_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=10000)
_rag_document_matrix = _rag_vectorizer.fit_transform(_rag_documents_text)


def _retrieve_evidence(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
    query_vec = _rag_vectorizer.transform([query])
    sims = _cos_sim(query_vec, _rag_document_matrix).flatten()
    ranked = sims.argsort()[::-1][:top_k]
    return [{
        "doc_id": rag_knowledge_documents[i]["doc_id"],
        "title": rag_knowledge_documents[i]["title"],
        "source": rag_knowledge_documents[i]["source"],
        "content": rag_knowledge_documents[i]["content"],
        "similarity_score": round(float(sims[i]), 4),
    } for i in ranked]


def answer_rag_query(query: str, top_k: int = 3) -> Dict[str, Any]:
    if _gemini_client is None:
        raise RuntimeError("GEMINI_API_KEY not configured -- cannot answer RAG queries.")

    retrieved = _retrieve_evidence(query, top_k)
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

    response = _gemini_client.models.generate_content(model=GEMINI_MODEL_NAME, contents=prompt)
    answer = getattr(response, "text", None) or "No response generated."
    return {"query": query, "answer": answer.strip(), "retrieved_evidence": retrieved}


# ------------------------------------------------------------
# MODULE 8: EXPLAINABLE AI (static, pre-generated)
# ------------------------------------------------------------
with open(XAI_DIR / "reports" / "churn_explanation.json", "r", encoding="utf-8") as f:
    churn_explanation = json.load(f)

with open(XAI_DIR / "reports" / "recommendation_explanation.json", "r", encoding="utf-8") as f:
    recommendation_explanation = json.load(f)

# ------------------------------------------------------------
# FASTAPI APPLICATION
# ------------------------------------------------------------
app = FastAPI(
    title="STREAMINTEL 360 Integration API",
    description="Unified API for churn, forecasting, recommendations, computer vision, sentiment, LLM intelligence, RAG, and explainability.",
    version="1.0.0",
)


class ChurnRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Subscriber feature values; missing columns default to 0")


class ForecastRequest(BaseModel):
    recent_hours: List[Dict[str, float]] = Field(..., description=f"Exactly {FORECAST_WINDOW} hourly rows of engineered features")


class RecommendRequest(BaseModel):
    user_id: Optional[str] = None
    seed_movie_id: Optional[str] = None
    top_n: int = 10


class SentimentRequest(BaseModel):
    reviews: Union[str, List[str]]


class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = 3


@app.get("/health")
def health():
    return {
        "status": "ok",
        "project": "STREAMINTEL 360",
        "modules_loaded": [
            "churn", "forecasting", "recommendations", "computer_vision",
            "sentiment", "llm_intelligence", "rag", "explainable_ai",
        ],
    }


@app.post("/churn/predict")
def churn_predict(request: ChurnRequest):
    try:
        return predict_churn(request.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/forecast/predict")
def forecast_predict(request: ForecastRequest):
    try:
        return predict_demand(request.recent_hours)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/recommend")
def recommend(request: RecommendRequest):
    try:
        return {"recommendations": get_recommendations(request.user_id, request.seed_movie_id, request.top_n)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/vision/predict")
async def vision_predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        return predict_genres(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/sentiment/predict")
def sentiment_predict(request: SentimentRequest):
    try:
        reviews = [request.reviews] if isinstance(request.reviews, str) else request.reviews
        return {"predictions": predict_sentiment(reviews)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/intelligence/summary")
def intelligence_summary():
    return executive_report


@app.post("/rag/query")
def rag_query(request: RAGQueryRequest):
    try:
        return answer_rag_query(request.query, request.top_k)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/explainability/churn")
def explainability_churn():
    return churn_explanation


@app.get("/explainability/recommendation")
def explainability_recommendation():
    return recommendation_explanation
