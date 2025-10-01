from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "tahsin/bangla-sentiment-analysis-bert-base-multilingual"

print(f"Loading Bengali model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

print("Model and tokenizer loaded successfully!")

sentiment_pipeline = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer
)
print("Pipeline created successfully!")

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze_text(request: TextInput):
    result = sentiment_pipeline(request.text)
    return {"input_text": request.text, "sentiment_analysis": result}

@app.get("/")
def read_root():
    return {"message": "Bengali Sentiment Analysis API on Render is running!"}
