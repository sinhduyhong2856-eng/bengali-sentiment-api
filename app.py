from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- বিশেষজ্ঞ দলের সদস্যদের লোড করা হচ্ছে ---

print("Loading Specialist Team Models...")

# বিশেষজ্ঞ ১: Sentiment Analysis-এর জন্য
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
sentiment_pipeline = pipeline(
    task="sentiment-analysis",
    model=SENTIMENT_MODEL_NAME
)
print("Sentiment specialist is ready!")

# বিশেষজ্ঞ ২: Intent Detection-এর জন্য
INTENT_MODEL_NAME = "facebook/bart-large-mnli"
intent_pipeline = pipeline(
    task="zero-shot-classification",
    model=INTENT_MODEL_NAME
)
print("Intent specialist is ready!")

print("Specialist Team successfully assembled!")

# --- FastAPI অ্যাপ ---
app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze_text(request: TextInput):
    # Intent-এর জন্য আমরা কী কী খুঁজতে চাই, তার একটি তালিকা
    intent_labels = [
        "product inquiry", "price question", "delivery issue", 
        "complaint", "gratitude", "order placement", "general conversation"
    ]

    # দুইজন বিশেষজ্ঞকেই তাদের কাজ দেওয়া হচ্ছে
    sentiment_result = sentiment_pipeline(request.text)
    intent_result = intent_pipeline(request.text, candidate_labels=intent_labels)

    # দুজনের ফলাফল একত্রিত করে চূড়ান্ত রিপোর্ট তৈরি
    return {
        "input_text": request.text,
        "sentiment_analysis": sentiment_result[0],
        "intent_analysis": {
            "main_intent": intent_result['labels'][0],
            "all_scores": dict(zip(intent_result['labels'], intent_result['scores']))
        }
    }

@app.get("/")
def read_root():
    return {"message": "Specialist Team Analysis API is running!"}
