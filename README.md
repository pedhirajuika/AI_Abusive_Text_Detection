# AI-Based Abusive & Cyberbullying Text Detection System

## Project Overview

This project is an AI-based text detection system that identifies abusive, offensive, hate, and potentially harmful text.

The system uses Natural Language Processing (NLP) and Machine Learning to classify text and provide:

- Normal / Abusive-Hate classification
- Severity level
- Confidence score
- Recommended action
- Automatic report storage

## Features

- English text detection
- Telugu/Tanglish dataset support
- Text preprocessing and cleaning
- Word-level TF-IDF features
- Character-level TF-IDF features
- Machine Learning classification
- Real-time prediction through FastAPI
- Automatic report generation
- SQLite database for report storage
- Web-based user interface
- Report history with search/filter
- Deployed online

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- FastAPI
- Uvicorn
- SQLite
- HTML
- CSS
- JavaScript
- GitHub

## Machine Learning Model

The system uses a combination of:

- Word-level TF-IDF
- Character-level TF-IDF
- Linear SVM
- CalibratedClassifierCV

The combined model achieved approximately *76% accuracy* on the test dataset.

## Dataset

The project uses:

- OLID / OffensEval dataset for English
- Telugu/Tanglish hate-speech dataset

The combined dataset contains approximately *17,240 text samples*.

## System Workflow

User Text  
↓  
Text Preprocessing  
↓  
Word + Character TF-IDF  
↓  
Machine Learning Model  
↓  
Classification  
↓  
Severity & Confidence  
↓  
Alert / Report  
↓  
SQLite Database

## API

The application provides a FastAPI REST API.

### Prediction Endpoint

POST /predict

Example request:

```json
{
  "text": "You are a disgusting idiot"
}
