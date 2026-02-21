from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from pathlib import Path
import json
import uuid
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.services.email_topic_inference import EmailTopicInferenceService
from app.dataclasses import Email
from app.features.factory import FeatureGeneratorFactory

router = APIRouter()

# -----------------------------
# Simple file persistence
# -----------------------------
TOPICS_PATH = Path("data/topic_keywords.json")
EMAILS_PATH = Path("data/stored_emails.json")

def read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# -----------------------------
# Models
# -----------------------------
class EmailRequest(BaseModel):
    subject: str
    body: str
    mode: str = Field("topic", pattern="^(topic|email)$")  # NEW

class TopicCreateRequest(BaseModel):
    name: str
    description: str

class EmailStoreRequest(BaseModel):
    subject: str
    body: str
    ground_truth: Optional[str] = None  # NEW

class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str

class EmailClassificationResponse(BaseModel):
    predicted_topic: Optional[str]
    topic_scores: Dict[str, float]
    features: Dict[str, Any]
    available_topics: List[str]

class EmailAddResponse(BaseModel):
    message: str
    email_id: str   # changed from int -> str (uuid)

# -----------------------------
# 1) Add topics dynamically
# -----------------------------
@router.post("/topics")
async def add_topic(request: TopicCreateRequest):
    try:
        topics = read_json(TOPICS_PATH, default={})
        topics[request.name] = {"description": request.description}  # ✅ FIX
        write_json(TOPICS_PATH, topics)
        return {"message": "Topic added", "topics": topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# 2) Store emails (with optional ground truth)
# -----------------------------
@router.post("/emails", response_model=EmailAddResponse)
async def store_email(request: EmailStoreRequest):
    """
    Store an email + optional ground_truth to data/stored_emails.json.
    Uses EmailTopicInferenceService to generate features (including embedding),
    so we don't depend on FeatureGeneratorFactory signature here.
    """
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)

        # Reuse existing pipeline to compute features (including embedding)
        result = inference_service.classify_email(email)
        features = result.get("features", {})

        embedding = features.get("email_embeddings_average_embedding")
        if embedding is None:
            raise ValueError("Missing feature key: email_embeddings_average_embedding")

        emails = read_json(EMAILS_PATH, default=[])
        new_id = str(uuid.uuid4())

        record = {
            "id": new_id,
            "subject": request.subject,
            "body": request.body,
            "ground_truth": request.ground_truth,
            "embedding": embedding,
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        emails.append(record)
        write_json(EMAILS_PATH, emails)

        return EmailAddResponse(message="Email stored", email_id=new_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: list stored emails for demo/debug
@router.get("/emails")
async def list_emails():
    try:
        return {"emails": read_json(EMAILS_PATH, default=[])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# 3) Classify: topic mode OR email mode
# -----------------------------
@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(request: EmailRequest):
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
        result = inference_service.classify_email(email)

        # default: topic mode (existing behavior)
        if request.mode == "topic":
            return EmailClassificationResponse(
                predicted_topic=result["predicted_topic"],
                topic_scores=result["topic_scores"],
                features=result["features"],
                available_topics=result["available_topics"]
            )

        # email mode: nearest stored email by embedding
        stored = read_json(EMAILS_PATH, default=[])
        if not stored:
            return EmailClassificationResponse(
                predicted_topic=None,
                topic_scores={},
                features=result["features"],
                available_topics=result["available_topics"]
            )

        incoming_emb = result["features"].get("email_embeddings_average_embedding")
        if incoming_emb is None:
            raise ValueError("Missing feature key: email_embeddings_average_embedding")

        incoming_emb = np.array(incoming_emb, dtype=float)
        stored_embeddings = np.array([e["embedding"] for e in stored], dtype=float)

        sims = cosine_similarity([incoming_emb], stored_embeddings)[0]
        best_idx = int(np.argmax(sims))
        best = stored[best_idx]

        predicted = best.get("ground_truth") or "unknown"

        return EmailClassificationResponse(
            predicted_topic=predicted,
            topic_scores={},  # not used in email mode
            features=result["features"],
            available_topics=result["available_topics"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topics")
async def topics():
    """Get available email topics"""
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"]}

@router.get("/pipeline/info")
async def pipeline_info():
    inference_service = EmailTopicInferenceService()
    return inference_service.get_pipeline_info()

# TODO: /features endpoint can be done later