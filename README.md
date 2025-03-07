project:
  name: "Predicting Diabetes Using Machine Learning"
  description: "A Data Mining Principles project focused on predicting diabetes using various machine learning models, including Random Forest, XGBoost, Logistic Regression, and KNN."
  author: "Group 1139 (Ariton Alexandru, Bucur Alexia-Gabriela, Coman Alex, Cojocaru Florin)"
  institution: "Bucharest University of Economic Studies"
  year: 2025

setup:
  dependencies:
    - R
    - caret
    - tidyverse
    - MASS
    - mlbench
    - summarytools
    - corrplot
    - gridExtra
    - timeDate
    - pROC
    - caTools
    - rpart.plot
    - e1071
    - graphics
  dataset:
    source: "Pima Indians Diabetes Dataset (Kaggle)"
    features:
      - Pregnant
      - Glucose
      - Pressure
      - Triceps
      - Insulin
      - Mass (BMI)
      - Pedigree
      - Age
      - Diabetes (Target Variable)
  installation:
    steps:
      - "Install required R packages using install.packages()"
      - "Download the dataset and place it in the working directory"
      - "Run the R script (cod.R) to execute the analysis"

data_analysis:
  preprocessing:
    - Convert diabetes column to factor (pos/neg)
    - Split dataset into training (70%) and testing (30%)
    - Handle missing values and outliers
  visualization:
    - Generate univariate and bivariate analysis plots
    - Compute correlation matrix and visualize with corrplot
  models:
    - Random Forest
    - XGBoost
    - K-Nearest Neighbors (KNN)
    - Logistic Regression
  evaluation:
    - Sensitivity, Specificity, Precision, Recall, F1 Score
    - ROC Curve and AUC
    - Confusion Matrix

usage:
  commands:
    - "source('cod.R') # Runs the full pipeline"
  expected_output:
    - Model comparison plots
    - Performance metrics
    - ROC curves and confusion matrices

conclusion:
  key_findings:
    - Logistic Regression outperformed other models
    - Glucose and BMI were the strongest predictors
    - The dataset showed imbalance in some variables
  future_work:
    - Try deep learning models (e.g., neural networks)
    - Improve feature engineering
    - Address class imbalance with SMOTE

license:
  type: "MIT"
  details: "This project is open-source and can be modified and shared under the MIT license."

contact:
  email: "your-email@example.com"
  contributors:
    - "Ariton Alexandru"
    - "Bucur Alexia-Gabriela"
    - "Coman Alex"
    - "Cojocaru Florin"
