# 🚗 Automobile Price Prediction using Decision Tree Regression

## 📌 Project Overview

This project implements a Decision Tree Regression model to predict
automobile prices using the Auto Dataset. The project includes data
preprocessing, feature selection, model training, evaluation, and
hyperparameter tuning.

## 🎯 Objectives

-   Perform data cleaning and preprocessing.
-   Convert categorical data into numerical format.
-   Handle missing values.
-   Train a Decision Tree Regression model.
-   Evaluate model performance using regression metrics.
-   Optimize the model using hyperparameter tuning.

## 📂 Dataset

**Dataset Name:** Auto Dataset

The dataset contains automobile features such as: - Engine
specifications - Horsepower - Number of cylinders - Vehicle dimensions -
Mileage information - Automobile price

**Target Variable:** Price

## 🛠️ Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Matplotlib
-   Seaborn
-   Jupyter Notebook

## 🔄 Project Workflow

### 1. Data Loading

The automobile dataset is loaded using Pandas.

### 2. Data Preprocessing

Performed: - Missing value handling - Data type conversion - Categorical
value encoding - Feature preparation

### 3. Model Building

A Decision Tree Regressor is trained to predict automobile prices.

### 4. Model Evaluation

Performance is evaluated using: - Mean Absolute Error (MAE) - Mean
Squared Error (MSE) - Root Mean Squared Error (RMSE) - R² Score

### 5. Hyperparameter Tuning

GridSearchCV is used to find the best Decision Tree parameters and
improve performance.

## 📁 Project Structure

    Auto-Price-Prediction/
    │
    ├── autos_dataset.csv
    ├── Day 60_Decision Tree Linear regression_Auto Data Set.ipynb
    ├── README.md
    └── requirements.txt

## 🚀 How to Run

1.  Clone the repository

```{=html}
<!-- -->
```
    git clone <repository-link>

2.  Install required libraries

```{=html}
<!-- -->
```
    pip install -r requirements.txt

3.  Open Jupyter Notebook

```{=html}
<!-- -->
```
    jupyter notebook

Run the notebook file:

`Day 60_Decision Tree Linear regression_Auto Data Set.ipynb`

## 📌 Future Improvements

-   Apply Random Forest Regression.
-   Use Gradient Boosting algorithms.
-   Perform feature importance analysis.
-   Deploy using Streamlit.

## 👩‍💻 Author

**Aditi Gawade**\
AI & Data Science Engineering Student

Skills: - Python - Machine Learning - Data Science - SQL

## ⭐ Conclusion

This project demonstrates the complete machine learning workflow for
automobile price prediction using Decision Tree Regression, including
preprocessing, training, evaluation, and optimization.
