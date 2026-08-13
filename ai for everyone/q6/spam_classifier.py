# Spam Email Classification using TF-IDF and Naive Bayes

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -----------------------------
# Load Dataset
# -----------------------------
# The UCI dataset is tab-separated, not CSV.
df = pd.read_csv(
    "/Users/suriya/Documents/ai for everyone/q6/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "text"]
)

print("First 5 Rows:")
print(df.head())

# Convert labels into numbers
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Features and Target
X = df["text"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(stop_words="english")

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Print Evaluation Report
print("\nClassification Report\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"]
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot(cmap="Blues")
plt.title("Spam Email Classification")
plt.savefig("confusion_matrix.png")
plt.show()
