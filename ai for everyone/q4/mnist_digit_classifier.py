import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------
# Load Dataset
# ------------------------------------------------

digits = load_digits()

X = digits.data
y = digits.target
images = digits.images

print("=" * 60)
print("DIGITS DATASET")
print("=" * 60)
print("Total Samples :", len(X))
print("Number of Features :", X.shape[1])

# ------------------------------------------------
# Train-Test Split
# ------------------------------------------------

X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X,
    y,
    images,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------
# Train Logistic Regression
# ------------------------------------------------

model = LogisticRegression(
    max_iter=3000,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed.")

# ------------------------------------------------
# Prediction
# ------------------------------------------------

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print("\nTraining Accuracy : {:.2f}%".format(train_accuracy * 100))
print("Testing Accuracy  : {:.2f}%".format(test_accuracy * 100))

# ------------------------------------------------
# Save Observations
# ------------------------------------------------

with open("observations.txt", "w") as file:

    file.write("MNIST Digit Recognition\n\n")

    file.write(f"Training Accuracy : {train_accuracy*100:.2f}%\n")

    file.write(f"Testing Accuracy : {test_accuracy*100:.2f}%\n")

# ------------------------------------------------
# Correct Predictions
# ------------------------------------------------

correct = []

for i in range(len(y_test)):
    if y_test[i] == test_pred[i]:
        correct.append(i)

plt.figure(figsize=(10,4))

for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(img_test[correct[i]], cmap="gray")
    plt.title(f"P:{test_pred[correct[i]]}")
    plt.axis("off")

plt.suptitle("Five Correct Predictions")
plt.tight_layout()

plt.savefig("correct_predictions.png", dpi=300)

plt.show()

# ------------------------------------------------
# Incorrect Predictions
# ------------------------------------------------

incorrect = []

for i in range(len(y_test)):
    if y_test[i] != test_pred[i]:
        incorrect.append(i)

plt.figure(figsize=(10,4))

count = min(5, len(incorrect))

for i in range(count):
    plt.subplot(1,5,i+1)
    plt.imshow(img_test[incorrect[i]], cmap="gray")
    plt.title(
        f"T:{y_test[incorrect[i]]}\nP:{test_pred[incorrect[i]]}"
    )
    plt.axis("off")

plt.suptitle("Incorrect Predictions")
plt.tight_layout()

plt.savefig("incorrect_predictions.png", dpi=300)

plt.show()

print("\nFiles Generated")
print("--------------------------")
print("correct_predictions.png")
print("incorrect_predictions.png")
print("observations.txt")
