from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# একটি জনপ্রিয় এবং পরীক্ষিত বহুভাষিক মডেল ব্যবহার করা হচ্ছে যা বাংলা সাপোর্ট করে
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"

print(f"Loading the final, reliable model: {MODEL_NAME}")

# ধাপ ১: টোকেনাইজার ম্যানুয়ালি লোড করা
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ধাপ ২: মডেল ম্যানুয়ালি লোড করা
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

print("Model and tokenizer loaded successfully!")

# ধাপ ৩: পাইপলাইন তৈরি করা
sentiment_pipeline = pipeline(
    task="sentiment-analysis",
    model=model,
    tokenizer=tokenizer
)
print("Pipeline created successfully!")

# --- FastAPI অ্যাপের বাকি অংশ ---
app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze_text(request: TextInput):
    result = sentiment_pipeline(request.text)
    return {"input_text": request.text, "sentiment_analysis": result}

@app.get("/")
def read_root():
    return {"message": "Multilingual Sentiment Analysis API is running!"}
