# Heart Disease Prediction using Machine Learning

## Objective

The objective of this project is to predict whether a patient has heart disease using Machine Learning algorithms.

The project demonstrates the complete Machine Learning workflow including:

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Feature Engineering
- Logistic Regression
- Random Forest
- Model Comparison
- Deployment Recommendation

---

## Dataset

Dataset Name:
Heart Disease Dataset (UCI)

Input File:

heart.csv

Target Column:

num

The target variable is converted into a binary classification problem.

- num = 0  → No Heart Disease
- num > 0 → Heart Disease

---

## Technologies Used

- Python 3.11+
- Pandas
- Matplotlib
- Scikit-learn

---

## Required Libraries

Install the required packages using:

```bash
pip install pandas matplotlib scikit-learn
```

---

## Project Structure

```
Question5/
│
├── heart.csv
├── heart_disease_prediction.py
├── cleaned_heart.csv
├── age_distribution.png
├── target_distribution.png
├── gender_distribution.png
├── model_comparison.png
├── evaluation_metrics.txt
├── README.md
└── report.pdf
```

---

## Steps Performed

### 1. Load Dataset

The Heart Disease dataset is loaded using Pandas.

---

### 2. Exploratory Data Analysis (EDA)

EDA is performed to understand the dataset before training.

The following analyses are performed:

- Display first five rows
- Dataset shape
- Dataset information
- Missing values
- Summary statistics

The following plots are generated:

- Age Distribution
- Heart Disease Distribution
- Gender Distribution

---

### 3. Data Cleaning

The following cleaning operations are performed:

- Remove duplicate records
- Handle missing values
- Save cleaned dataset

---

### 4. Feature Engineering

Categorical variables are converted into numerical values using Label Encoding.

The target variable is converted into binary form:

- 0 → No Heart Disease
- 1 → Heart Disease

---

### 5. Model Training

Two Machine Learning models are trained.

- Logistic Regression
- Random Forest

---

### 6. Model Evaluation

The following metrics are calculated:

- Accuracy
- Precision
- Recall
- F1-score

The evaluation results are saved in:

evaluation_metrics.txt

---

### 7. Model Comparison

A bar chart is generated to compare the accuracies of:

- Logistic Regression
- Random Forest

The graph is saved as:

model_comparison.png

---

### 8. Deployment Recommendation

The model with the highest accuracy is recommended for deployment.

---

## Output Files

The project generates the following files automatically.

- cleaned_heart.csv
- age_distribution.png
- target_distribution.png
- gender_distribution.png
- model_comparison.png
- evaluation_metrics.txt

---

## How to Run

Open Terminal inside the project folder.

Run:

```bash
python heart_disease_prediction.py
```

The program will:

- Load the dataset
- Perform EDA
- Clean the dataset
- Perform Feature Engineering
- Train both models
- Compare their performance
- Generate plots
- Save evaluation metrics
- Recommend the best model

---

## Conclusion

This project demonstrates an end-to-end Machine Learning pipeline for predicting heart disease.

Among the two models, the model with the higher accuracy is recommended for deployment.
