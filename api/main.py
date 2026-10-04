import sys
import os
import sqlite3

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.prediction.predict import detect_text
from database.database import get_reports

app = FastAPI(
    title="AI Abusive Text Detection API",
    description="API for detecting abusive and hate text",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class TextRequest(BaseModel):
    text: str


def save_report(result, text):
    connection = sqlite3.connect("database/abusive_text.db")
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO reports
        (text, category, severity, confidence, action)
        VALUES (?, ?, ?, ?, ?)
    """, (
        text,
        result["category"],
        result["severity"],
        result["confidence"],
        result["action"]
    ))

    connection.commit()
    connection.close()


@app.get("/")
def home():
    return {
        "message": "AI Abusive Text Detection API is running"
    }


@app.post("/predict")
def predict(request: TextRequest):
    result = detect_text(request.text)

    save_report(result, request.text)

    return result


@app.get("/reports")
def reports():
    data = get_reports()

    return {
        "reports": [
            {
                "id": row[0],
                "text": row[1],
                "category": row[2],
                "severity": row[3],
                "confidence": row[4],
                "action": row[5]
            }
            for row in data
        ]
    }