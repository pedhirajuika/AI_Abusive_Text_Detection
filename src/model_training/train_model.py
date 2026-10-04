import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack

data = pd.read_csv(
    "data/processed/combined_dataset.csv"
)

data = data.dropna(
    subset=["clean_text", "target"]
)

X = data["clean_text"]
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=80000,
    sublinear_tf=True
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)

X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char = char_vectorizer.transform(X_test)

X_train_combined = hstack([
    X_train_word,
    X_train_char
])

X_test_combined = hstack([
    X_test_word,
    X_test_char
])

base_model = LinearSVC(
    class_weight="balanced"
)

model = CalibratedClassifierCV(
    base_model,
    cv=3
)

model.fit(
    X_train_combined,
    y_train
)

y_pred = model.predict(
    X_test_combined
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================")
print("WORD + CHARACTER SVM")
print("==============================")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Abusive/Hate"
        ]
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

joblib.dump(
    model,
    "models/abusive_text_model.pkl"
)

joblib.dump(
    word_vectorizer,
    "models/word_vectorizer.pkl"
)

joblib.dump(
    char_vectorizer,
    "models/char_vectorizer.pkl"
)

print("\nWord + Character SVM model saved successfully!")