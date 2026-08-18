# Machine Learning - Assignment 2
**Student ID**: 2025AC05718 
---

## a. Problem Statement

The objective of this assignment is to implement, evaluate, and deploy multiple machine learning classification models on a real-world tabular dataset. The task is to predict the likelihood of **Heart Disease** in patients using 13 clinical diagnostic features such as chest pain type, resting blood pressure, cholesterol level, maximum heart rate achieved, and exercise-induced ST depression.

Five different classifiers (Logistic Regression, Decision Tree, k-Nearest Neighbors, Naive Bayes, and Random Forest Ensemble) were trained and compared using six evaluation metrics: Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). The models are demonstrated through an interactive Streamlit web application.

---

## b. Dataset Description

- **Dataset**: Heart Disease Classification Dataset (UCI Repository standard)
- **Total Records**: 1,025 instances (satisfies the minimum requirement of 500 instances)
- **Total Features**: 13 input features + 1 binary target (satisfies the minimum requirement of 12 features)

### Attribute Information:
1. `age`: Age in years (range: 29 - 77)
2. `sex`: Gender (1 = male, 0 = female)
3. `cp`: Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)
4. `trestbps`: Resting blood pressure in mm Hg (range: 94 - 200)
5. `chol`: Serum cholesterol in mg/dl (range: 126 - 564)
6. `fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. `restecg`: Resting electrocardiographic results (0, 1, 2)
8. `thalach`: Maximum heart rate achieved (range: 71 - 202)
9. `exang`: Exercise induced angina (1 = yes, 0 = no)
10. `oldpeak`: ST depression induced by exercise relative to rest (range: 0.0 - 6.2)
11. `slope`: Slope of peak exercise ST segment (0, 1, 2)
12. `ca`: Number of major vessels colored by flourosopy (0 - 4)
13. `thal`: Thalassemia category (0, 1, 2, 3)
14. `target`: Diagnosis of heart disease (0 = absence, 1 = presence)

---

## c. GitHub Repository Link

- **GitHub Repository**: `https://github.com/2025ac05718/My_ML_Assignment_02`
- **Live Streamlit App**: `https://mymlassignment02-mydqfpameldrsyqm9lrvla.streamlit.app`

---

## d. Models Used & Comparison Table

The models were evaluated on an 80-20 stratified train-test split (205 test instances):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9951** | **1.0000** | **1.0000** | **0.9643** | **0.9818** | **0.9792** |
| **Decision Tree** | 0.9073 | 0.7336 | 0.7143 | 0.5357 | 0.6122 | 0.5683 |
| **kNN** | 0.9268 | 0.9459 | 0.8824 | 0.5357 | 0.6667 | 0.6530 |
| **Naive Bayes** | 0.9268 | 0.9514 | 0.8824 | 0.5357 | 0.6667 | 0.6530 |
| **Random Forest (Ensemble)** | 0.9122 | 0.9679 | 1.0000 | 0.3571 | 0.5263 | 0.5694 |

---

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Logistic Regression demonstrated the best performance overall, reaching 99.51% Accuracy and an MCC of 0.9792. The continuous linear relationship of clinical risk factors provides a clean separating hyperplane. |
| **Decision Tree** | The Decision Tree model achieved 90.73% Accuracy. Due to its axis-aligned splitting structure, it exhibited moderate recall (53.57%) on border cases. |
| **kNN** | kNN reached 92.68% Accuracy and 0.9459 AUC after standardizing numeric attributes using `StandardScaler`, showing consistent local neighbor clustering. |
| **Naive Bayes** | Gaussian Naive Bayes performed strongly with 92.68% Accuracy and 0.9514 AUC. The independent likelihood assumption works effectively for these independent clinical tests. |
| **Random Forest (Ensemble)** | Random Forest obtained 100% Precision (zero false positives) and 0.9679 AUC. However, its conservative tree voting resulted in lower recall on this test split. |
| **Overall Winner for your dataset?** | **Logistic Regression** is the overall best model for this dataset based on superior Accuracy (0.9951), AUC (1.0000), F1 Score (0.9818), and MCC (0.9792). |

---

## Repository Structure

```text
.
├── app.py                     # Streamlit application
├── requirements.txt           # Deployment dependencies
├── README.md                  # Assignment documentation
├── test_data.csv              # Hold-out test dataset
├── heart_disease_dataset.csv  # Full dataset (1025 samples)
├── train_model.py             # Model training script
└── model/                     # Serialized model files
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── scaler.pkl
    └── metrics_summary.json
```

---

## Execution Instructions

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Train models (optional):
   ```bash
   python train_model.py
   ```
3. Launch Streamlit web app:
   ```bash
   python -m streamlit run app.py
   ```
