import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------
# Load Dataset
# ----------------------------------------

print("Loading Dataset...")

df = pd.read_csv("dataset/food_delivery.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# ----------------------------------------
# Data Cleaning
# ----------------------------------------

print("\nChecking Missing Values...")

print(df.isnull().sum())

# Remove missing values
df.dropna(inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Remove Order_ID (not useful)
df.drop("Order_ID", axis=1, inplace=True)

print("\nDataset Shape After Cleaning:", df.shape)

# ----------------------------------------
# Data Visualization
# ----------------------------------------

# Delivery Time Distribution
plt.figure(figsize=(6,4))
plt.hist(df["Delivery_Time_min"], bins=20, edgecolor="black")
plt.title("Delivery Time Distribution")
plt.xlabel("Delivery Time (Minutes)")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig("delivery_time_distribution.png")
plt.close()

# Average Delivery Time by Weather
weather = df.groupby("Weather")["Delivery_Time_min"].mean()

plt.figure(figsize=(6,4))
weather.plot(kind="bar")
plt.title("Average Delivery Time by Weather")
plt.xlabel("Weather")
plt.ylabel("Average Delivery Time")
plt.tight_layout()
plt.savefig("weather_delivery_time.png")
plt.close()

# Average Delivery Time by Traffic
traffic = df.groupby("Traffic_Level")["Delivery_Time_min"].mean()

plt.figure(figsize=(6,4))
traffic.plot(kind="bar")
plt.title("Average Delivery Time by Traffic")
plt.xlabel("Traffic Level")
plt.ylabel("Average Delivery Time")
plt.tight_layout()
plt.savefig("traffic_delivery_time.png")
plt.close()

print("\nCharts Saved Successfully.")

# ----------------------------------------
# Data Preprocessing
# ----------------------------------------

encoder = LabelEncoder()

categorical_columns = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type"
]

for column in categorical_columns:
    df[column] = encoder.fit_transform(df[column].astype(str))

print("\nEncoded Data Types:")
print(df.dtypes)

# ----------------------------------------
# Features and Target
# ----------------------------------------

X = df.drop("Delivery_Time_min", axis=1)
y = df["Delivery_Time_min"]

# ----------------------------------------
# Train-Test Split
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------------
# Model Training
# ----------------------------------------

print("\nTraining Model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------------------
# Prediction
# ----------------------------------------

predictions = model.predict(X_test)

# ----------------------------------------
# Evaluation
# ----------------------------------------

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n========== MODEL EVALUATION ==========")
print("Mean Absolute Error :", round(mae,2))
print("Root Mean Squared Error :", round(rmse,2))
print("R2 Score :", round(r2,2))

# ----------------------------------------
# Error Analysis
# ----------------------------------------

errors = abs(y_test - predictions)

print("\nAverage Prediction Error:",
      round(errors.mean(),2), "minutes")

# ----------------------------------------
# Feature Importance
# ----------------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=True)

plt.figure(figsize=(7,5))
importance.plot(kind="barh")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# ----------------------------------------
# Save Model
# ----------------------------------------

joblib.dump(model, "delivery_time_model.pkl")

print("\nModel Saved Successfully.")

print("\nProject Completed Successfully.")
