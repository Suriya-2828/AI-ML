# ==========================================================
# Heart Disease Prediction using Machine Learning
#
# Question 5
#
# Perform:
# 1. Exploratory Data Analysis (EDA)
# 2. Data Cleaning
# 3. Feature Engineering
# 4. Logistic Regression
# 5. Random Forest
# 6. Compare Both Models
#
# ==========================================================

# ==========================================================
# STEP 1 : Import Required Libraries
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================================
# STEP 2 : Load Dataset
# ==========================================================

print("="*60)
print("HEART DISEASE PREDICTION")
print("="*60)

df = pd.read_csv("heart.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

# ==========================================================
# STEP 3 : Exploratory Data Analysis (EDA)
#
# EDA is performed to understand the dataset before
# training the machine learning model.
# ==========================================================

print("\nMissing Values")
print(df.isnull().sum())

print("\nSummary Statistics")
print(df.describe())

# ----------------------------------------------------------
# Plot 1 : Age Distribution
# ----------------------------------------------------------

plt.figure(figsize=(7,5))

df["age"].hist(bins=20)

plt.title("Age Distribution")

plt.xlabel("Age")

plt.ylabel("Number of Patients")

plt.tight_layout()

plt.savefig("age_distribution.png")

plt.show()

# ----------------------------------------------------------
# Convert Target Variable
#
# num = 0  -> No Heart Disease
# num > 0  -> Heart Disease
# ----------------------------------------------------------

df["HeartDisease"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

# ----------------------------------------------------------
# Plot 2 : Heart Disease Distribution
# ----------------------------------------------------------

plt.figure(figsize=(6,4))

df["HeartDisease"].value_counts().plot(kind="bar")

plt.title("Target Distribution")

plt.xlabel("Heart Disease")

plt.ylabel("Count")

plt.tight_layout()

plt.savefig("target_distribution.png")

plt.show()

# ----------------------------------------------------------
# Plot 3 : Gender Distribution
# ----------------------------------------------------------

plt.figure(figsize=(6,4))

df["sex"].value_counts().plot(kind="bar")

plt.title("Gender Distribution")

plt.xlabel("Gender")

plt.ylabel("Count")

plt.tight_layout()

plt.savefig("gender_distribution.png")

plt.show()

# ==========================================================
# STEP 4 : Data Cleaning
#
# Data cleaning removes duplicates and fills missing values.
# ==========================================================

print("\nRemoving Duplicate Rows...")

df.drop_duplicates(inplace=True)

print("Duplicates Removed.")

print("\nHandling Missing Values...")

for column in df.columns:

    if df[column].dtype == "object":

        df[column]=df[column].fillna(df[column].mode()[0])

    else:

        df[column]=df[column].fillna(df[column].median())

print("Missing Values Handled.")

# ==========================================================
# STEP 5 : Feature Engineering
#
# Machine Learning models cannot understand text.
# Convert categorical columns into numerical values.
# ==========================================================

print("\nEncoding Categorical Columns...")

label_encoder = LabelEncoder()

categorical_columns = [

    "sex",

    "dataset",

    "cp",

    "fbs",

    "restecg",

    "exang",

    "slope",

    "thal"

]

for column in categorical_columns:

    df[column] = label_encoder.fit_transform(df[column])

print("Encoding Completed.")

# ----------------------------------------------------------
# Remove Unnecessary Columns
# ----------------------------------------------------------

df.drop(columns=["id", "num"], inplace=True)

# ----------------------------------------------------------
# Save Cleaned Dataset
# ----------------------------------------------------------

df.to_csv("cleaned_heart.csv", index=False)

print("\nCleaned Dataset Saved Successfully.")

# ==========================================================
# STEP 6 : Prepare Features and Target Variable
# ==========================================================

X = df.drop("HeartDisease", axis=1)

y = df["HeartDisease"]

# ==========================================================
# STEP 7 : Split Dataset
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

# ==========================================================
# STEP 8 : Train Logistic Regression Model
# ==========================================================

print("\nTraining Logistic Regression...")

logistic_model = LogisticRegression(max_iter=1000)

logistic_model.fit(X_train, y_train)

logistic_prediction = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(

    y_test,

    logistic_prediction

)

print("\nLogistic Regression Accuracy :")

print(round(logistic_accuracy*100,2), "%")

# ==========================================================
# STEP 9 : Train Random Forest Model
# ==========================================================

print("\nTraining Random Forest...")

random_forest = RandomForestClassifier(

    random_state=42

)

random_forest.fit(

    X_train,

    y_train

)

rf_prediction = random_forest.predict(

    X_test

)

rf_accuracy = accuracy_score(

    y_test,

    rf_prediction

)

print("\nRandom Forest Accuracy :")

print(round(rf_accuracy*100,2), "%")

# ==========================================================
# STEP 10 : Compare Both Models
# ==========================================================

plt.figure(figsize=(7,5))

models = [

    "Logistic Regression",

    "Random Forest"

]

accuracy = [

    logistic_accuracy*100,

    rf_accuracy*100

]

plt.bar(

    models,

    accuracy

)

plt.ylabel("Accuracy (%)")

plt.title("Model Comparison")

plt.tight_layout()

plt.savefig("model_comparison.png")

plt.show()

# ==========================================================
# STEP 11 : Save Evaluation Metrics
# ==========================================================

with open(

    "evaluation_metrics.txt",

    "w"

) as file:

    file.write("HEART DISEASE PREDICTION\n\n")

    file.write(

        f"Logistic Regression Accuracy : {logistic_accuracy*100:.2f}%\n\n"

    )

    file.write(

        classification_report(

            y_test,

            logistic_prediction

        )

    )

    file.write("\n")

    file.write(

        f"Random Forest Accuracy : {rf_accuracy*100:.2f}%\n\n"

    )

    file.write(

        classification_report(

            y_test,

            rf_prediction

        )

    )

print("\nEvaluation Metrics Saved.")

# ==========================================================
# STEP 12 : Deployment Recommendation
# ==========================================================

print("\nRecommended Model")

if rf_accuracy > logistic_accuracy:

    print("Random Forest")

else:

    print("Logistic Regression")

print("\nProject Completed Successfully.")
