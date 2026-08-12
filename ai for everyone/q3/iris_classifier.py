# iris_classifier.py

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ----------------------------------------
# Load Dataset
# ----------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

print("=" * 60)
print("IRIS DATASET LOADED")
print("=" * 60)

print("Number of Samples :", len(X))
print("Number of Features:", X.shape[1])

# ----------------------------------------
# Split Dataset
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ----------------------------------------
# Train Decision Tree
# ----------------------------------------

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

print("\nDecision Tree Model Trained Successfully.")

# ----------------------------------------
# Prediction
# ----------------------------------------

y_pred = model.predict(X_test)

# ----------------------------------------
# Accuracy
# ----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy")
print(f"{accuracy*100:.2f}%")

# ----------------------------------------
# Confusion Matrix
# ----------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

# ----------------------------------------
# Classification Report
# ----------------------------------------

report = classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
)

print("\nClassification Report")
print(report)

# Save metrics

with open("evaluation_metrics.txt", "w") as file:

    file.write("Decision Tree Classification\n\n")

    file.write(f"Accuracy : {accuracy*100:.2f}%\n\n")

    file.write("Confusion Matrix\n")

    file.write(str(cm))

    file.write("\n\nClassification Report\n")

    file.write(report)

# ----------------------------------------
# Plot Confusion Matrix
# ----------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)

plt.show()

print("\nFiles Generated")
print("------------------------")
print("evaluation_metrics.txt")
print("confusion_matrix.png")
