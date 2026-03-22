# 📊 Bank Marketing Classification Project

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![LightGBM](https://img.shields.io/badge/LightGBM-GradientBoosting-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Project Overview

This project aims to predict whether a client will subscribe to a bank product based on marketing campaign data.

Due to strong **class imbalance**, the main focus is on optimizing the **F1-score for the positive class ("yes")**, balancing precision and recall.

---

## 🎯 Objectives

- Handle imbalanced data effectively  
- Build and compare multiple ML models  
- Optimize model performance using:
  - Feature Engineering
  - SMOTE
  - Class weighting
  - Threshold tuning  
- Select the best model based on **F1-score (Yes class)**  

---

## ⚙️ Tech Stack

- **Python 3.12**
- **Pandas / NumPy**
- **Scikit-learn**
- **Imbalanced-learn (SMOTE)**
- **XGBoost**
- **LightGBM**
___

| Model                                       | F1 Train (Yes) | F1 Test (Yes) | Precision / Recall Train (Yes) | Precision / Recall Test (Yes) |
| ------------------------------------------- | -------------- | ------------- | ------------------------------ | ----------------------------- |
| LogisticRegression (baseline) + SMOTE       | 0.44           | 0.45          | 0.34 / 0.63                    | 0.35 / 0.65                   |
| LogisticRegression (threshold=0.65) + SMOTE | 0.44           | 0.49          | 0.34 / 0.63                    | 0.44 / 0.56                   |
| RandomForest + SMOTE                        | 0.98           | 0.41          | 0.99 / 0.98                    | 0.51 / 0.34                   |
| KNN + SMOTE                                 | 0.54           | 0.34          | 0.37 / 0.98                    | 0.23 / 0.64                   |
| KNN (without SMOTE)                         | 0.47           | 0.36          | 0.73 / 0.35                    | 0.56 / 0.26                   |
| XGBoost + SMOTE                             | 0.49           | 0.41          | 0.69 / 0.38                    | 0.61 / 0.31                   |
| XGBoost + scale_pos_weight                  | 0.53           | 0.48          | 0.42 / 0.70                    | 0.39 / 0.62                   |
| LightGBM (Hyperopt)                         | 0.49           | 0.50          | 0.40 / 0.64                    | 0.40 / 0.64                   |
| LightGBM (RandomizedSearchCV)               | 0.50           | 0.50          | 0.41 / 0.65                    | 0.40 / 0.64                   |

---

## 🔧 Pipeline

```python
pipeline = ImbPipeline([
    ("feature_engineering", FunctionTransformer(feature_engineering)),
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", model)
])


