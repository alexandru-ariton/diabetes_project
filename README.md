# 📌 Predicting Diabetes Using Machine Learning

**A Data Mining Principles project focused on predicting diabetes using machine learning models.**  
This study applies various ML techniques to predict diabetes, using the **Pima Indians Diabetes Dataset**.

---

## 📂 Project Overview

- **Authors**: Ariton Alexandru, Bucur Alexia-Gabriela, Coman Alex, Cojocaru Florin  
- **Institution**: Bucharest University of Economic Studies  
- **Year**: 2025  
- **Dataset**: [Pima Indians Diabetes Dataset (Kaggle)](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

---

## 🛠 Setup & Dependencies

### 🔹 Required R Packages:
```r
install.packages(c("caret", "tidyverse", "MASS", "mlbench", "summarytools",
                   "corrplot", "gridExtra", "timeDate", "pROC", "caTools", 
                   "rpart.plot", "e1071", "graphics"))
```

### 🔹 Installation Steps:
1. Install the required R packages listed above.
2. Download the dataset and place it in the working directory.
3. Run the R script:
   ```r
   source("cod.R")
   ```

---

## 🔍 Data Analysis

### **Preprocessing**
✔ Convert diabetes column to **factor** (pos/neg)  
✔ Split dataset into **training (70%)** and **testing (30%)**  
✔ Handle **missing values** and **outliers**  

### **Visualization**
📊 Univariate and Bivariate Analysis  
📉 Correlation Matrix with `corrplot`  

### **Machine Learning Models**
✅ **Random Forest**  
✅ **XGBoost**  
✅ **K-Nearest Neighbors (KNN)**  
✅ **Logistic Regression**  

### **Performance Metrics**
📌 Sensitivity, Specificity, Precision, Recall, F1 Score  
📈 ROC Curve & AUC, Confusion Matrix  

---

## 🎯 Usage

To run the full pipeline:
```r
source("cod.R")
```

### **Expected Outputs**
✔ Model comparison plots  
✔ Performance metrics  
✔ ROC curves & confusion matrices  

---

## 📊 Results & Conclusion

### **Key Findings**
✔ **Logistic Regression** performed the best  
✔ **Glucose & BMI** are strong predictors of diabetes  
✔ The dataset contains some **imbalanced variables**  

### **Future Work**
🚀 Experiment with **Deep Learning** models (e.g., Neural Networks)  
⚙️ Improve **Feature Engineering**  
⚖️ Address **Class Imbalance** using **SMOTE**  

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 📩 Contact


👥 Contributors:  
- Ariton Alexandru  
- Bucur Alexia-Gabriela  
- Coman Alex  
- Cojocaru Florin  

---
