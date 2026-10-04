import sys
import os
import joblib
from scipy.sparse import hstack

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from utils.text_cleaner import clean_text

model = joblib.load(
    "models/abusive_text_model.pkl"
)

word_vectorizer = joblib.load(
    "models/word_vectorizer.pkl"
)

char_vectorizer = joblib.load(
    "models/char_vectorizer.pkl"
)


def detect_text(text):

    cleaned_text = clean_text(text)

    word_features = word_vectorizer.transform(
        [cleaned_text]
    )

    char_features = char_vectorizer.transform(
        [cleaned_text]
    )

    features = hstack([
        word_features,
        char_features
    ])

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    confidence = max(probabilities) * 100

    if prediction == 1:

        if confidence >= 90:
            severity = "High"
            action = "Immediate review recommended"

        elif confidence >= 70:
            severity = "Medium"
            action = "Review recommended"

        else:
            severity = "Low"
            action = "Monitor recommended"

        return {
            "status": "ALERT",
            "category": "Abusive/Hate",
            "severity": severity,
            "confidence": confidence,
            "action": action
        }

    return {
        "status": "SAFE",
        "category": "Normal",
        "severity": "Low",
        "confidence": confidence,
        "action": "No action required"
    }


if __name__ == "_main_":

    print("=" * 55)
    print("       AI ABUSIVE TEXT DETECTION SYSTEM")
    print("=" * 55)

    text = input("\nEnter a message: ")

    result = detect_text(text)

    print("\n" + "=" * 55)
    print("                 RESULT")
    print("=" * 55)

    if result["status"] == "ALERT":
        print("⚠️ ALERT: Potentially abusive content detected")
    else:
        print("✅ SAFE: No potentially abusive content detected")

    print(f"\nCategory   : {result['category']}")
    print(f"Severity   : {result['severity']}")
    print(f"Confidence : {result['confidence']:.2f}%")
    print(f"Action     : {result['action']}")

    print("=" * 55)