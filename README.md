# 🍷 Wine Quality Prediction with Linear Models (Regression & Classification)

This project applies machine learning fundamentals to predict wine quality based on physicochemical tests using regression and classification models. It is inspired by Chapter 4 of *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* and emphasizes model training, regularization, optimization, and evaluation.

## 📌 Objective

- Understand and apply core regression and classification algorithms.
- Explore regularization techniques (Ridge, Lasso).
- Compare batch vs stochastic gradient descent optimization.
- Perform hyperparameter tuning and learning curve analysis.
- Analyze bias-variance tradeoffs and error patterns.

## 🗃️ Dataset

- **Wine Quality Dataset**  
  - Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) or [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
  - Features: 11 physicochemical variables (e.g., pH, alcohol, sulphates)
  - Targets:
    - `quality` (score from 0–10, used for regression)
    - `high_quality` (binary label: 1 if quality ≥ 7, else 0 for classification)

## 🧠 Algorithms Implemented

- **Linear Regression (OLS)**
- **Stochastic Gradient Descent (SGD Regressor)**
- **Ridge & Lasso Regression (Regularization)**
- **Logistic Regression (for classification)**

## 🛠️ Pipeline Overview

1. **Data Loading & Cleaning**
   - Load `.csv` file using `pandas`
   - Handle missing values
   - Encode categorical variables if any

2. **Feature Engineering**
   - Polynomial Features (e.g., degree 2)
   - Scaling with `StandardScaler`

3. **Model Training**
   - Train OLS, Ridge, Lasso, and SGD regressors
   - Train Logistic Regression for binary classification

4. **Model Evaluation**
   - Regression: RMSE, R²
   - Classification: Accuracy, F1 Score, Confusion Matrix
   - Training time comparison

5. **Optimization Techniques**
   - Compare Batch Gradient Descent vs SGD convergence
   - Plot learning curves
   - GridSearchCV for hyperparameter tuning

6. **Error Analysis**
   - Identify overfitting or underfitting
   - Evaluate impact of polynomial degree on bias-variance tradeoff

## 📊 Results (Sample Output)

| Model  | RMSE | R²   | Time (s) |
|--------|------|------|----------|
| OLS    | 0.686 | 0.340 | 0.017 |
| Ridge  | 0.687 | 0.340 | 0.014 |
| Lasso  | 0.845 | 0.000 | 0.012 |
| SGD    | 0.688 | 0.337 | 0.015 |

## 🪄 Classification Metrics (Logistic Regression)

| Metric       | Score  |
|--------------|--------|
| Accuracy     | 89.3%  |
| F1 Score     | 0.84   |
| ROC AUC      | 0.91   |

## 📈 Learning Curves

- Plots show training vs validation performance as dataset size increases.
- Visual diagnosis of bias (underfitting) vs variance (overfitting).



