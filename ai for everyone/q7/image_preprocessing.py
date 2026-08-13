import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# -----------------------------
# Configuration
# -----------------------------
DATASET_PATH = "dataset"
OUTPUT_PATH = "output"

TRAIN_FOLDER = os.path.join(OUTPUT_PATH, "train")
VAL_FOLDER = os.path.join(OUTPUT_PATH, "validation")

IMAGE_SIZE = (128, 128)

os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(VAL_FOLDER, exist_ok=True)

metadata = []

# -----------------------------
# Read Images
# -----------------------------
print("Reading images...")

for label in os.listdir(DATASET_PATH):

    label_folder = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_folder):
        continue

    for image_name in os.listdir(label_folder):

        image_path = os.path.join(label_folder, image_name)

        metadata.append({
            "filepath": image_path,
            "label": label
        })

df = pd.DataFrame(metadata)

print(df.head())

# -----------------------------
# Train Validation Split
# -----------------------------
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# -----------------------------
# Function
# -----------------------------
def preprocess_and_save(dataframe, output_folder):

    for _, row in dataframe.iterrows():

        image = Image.open(row["filepath"]).convert("RGB")

        image = image.resize(IMAGE_SIZE)

        image_array = np.array(image).astype(np.float32) / 255.0

        image = Image.fromarray((image_array * 255).astype(np.uint8))

        class_folder = os.path.join(output_folder, row["label"])

        os.makedirs(class_folder, exist_ok=True)

        save_path = os.path.join(class_folder,
                                 os.path.basename(row["filepath"]))

        image.save(save_path)

# -----------------------------
# Process Images
# -----------------------------
print("Processing training images...")
preprocess_and_save(train_df, TRAIN_FOLDER)

print("Processing validation images...")
preprocess_and_save(val_df, VAL_FOLDER)

# -----------------------------
# Save Metadata
# -----------------------------
df.to_csv(os.path.join(OUTPUT_PATH, "metadata.csv"), index=False)

print("\nDone!")
print("Metadata saved.")
print("Training and Validation folders created.")
