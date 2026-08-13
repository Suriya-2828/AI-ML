# Food Delivery Time Prediction using Machine Learning

## 1. Introduction

Food delivery services need accurate delivery time predictions to improve customer satisfaction and operational efficiency. This project develops a machine learning model to estimate food delivery time based on order and delivery-related features.

---

## 2. Objective

- Predict food delivery time.
- Analyze factors affecting delivery duration.
- Generate useful business insights.

---

## 3. Dataset

The dataset contains information such as:

- Distance
- Weather
- Traffic Level
- Time of Day
- Vehicle Type
- Preparation Time
- Courier Experience

Target Variable:

- Delivery_Time_min

---

## 4. Data Cleaning

The following preprocessing steps were performed:

- Removed missing values
- Removed duplicate records
- Removed Order_ID column
- Encoded categorical variables using Label Encoding

---

## 5. Data Visualization

The following visualizations were created:

- Delivery Time Distribution
- Average Delivery Time by Weather
- Average Delivery Time by Traffic Level
- Feature Importance

---

## 6. Machine Learning Model

Model Used:

Random Forest Regressor

Reasons:

- Easy to implement
- Good prediction accuracy
- Handles both numerical and categorical features effectively

---

## 7. Evaluation

Evaluation Metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

These metrics help measure prediction accuracy and model performance.

---

## 8. Error Analysis

The model predicts delivery time accurately for most orders. Some prediction errors occur due to factors not included in the dataset, such as road accidents, sudden weather changes, or unexpected traffic congestion.

---

## 9. Business Insights

- Longer distances generally increase delivery time.
- Heavy traffic significantly delays deliveries.
- Poor weather conditions lead to longer delivery durations.
- Preparation time has a direct impact on total delivery time.
- Courier experience helps reduce delivery delays.

---

## 10. Conclusion

The project successfully demonstrates a complete machine learning pipeline for predicting food delivery time. The generated insights can help food delivery companies improve operational planning and provide more accurate delivery estimates to customers.
