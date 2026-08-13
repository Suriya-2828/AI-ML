import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATASET_PATH = "/Users/suriya/Documents/ai for everyone/q8/dataset"
IMAGE_SIZE = (64, 64)

images = []
labels = []

print("Loading images...")

for label in os.listdir(DATASET_PATH):

    folder = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        try:
            # Original image
            image = Image.open(path).convert("RGB")
            image = image.resize(IMAGE_SIZE)

            image_array = np.array(image) / 255.0

            images.append(image_array.flatten())
            labels.append(label)

            # -----------------------------
            # Data Augmentation
            # Horizontal Flip
            # -----------------------------
            flipped = image.transpose(Image.FLIP_LEFT_RIGHT)

            flipped_array = np.array(flipped) / 255.0

            images.append(flipped_array.flatten())
            labels.append(label)

        except:
            pass

X = np.array(images)
y = np.array(labels)

print("Total Images after Augmentation:", len(X))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report\n")
print(classification_report(y_test, predictions))

# Save Model
joblib.dump(model, "defect_model.pkl")

print("\nModel saved as defect_model.pkl")
