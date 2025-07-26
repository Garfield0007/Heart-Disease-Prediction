# ❤️ Heart Disease Prediction – Machine Learning Project

![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Executed on Jupyter Notebook](https://img.shields.io/badge/Executed%20on-Jupyter-orange?style=for-the-badge&logo=Jupyter&logoColor=white)
![Status: Completed](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

This repository contains my project on **Heart Disease Prediction**, where I used various classification algorithms to predict the likelihood of heart disease in patients based on health-related features. The goal is to leverage data and machine learning to support preventive healthcare.

---

## 🧠 Project Goal

To build and evaluate classification models that can predict the presence of heart disease based on medical and demographic attributes of patients.

---

## 📂 Dataset

- **Source:** [Kaggle – Heart Disease Prediction Dataset](https://www.kaggle.com/rishidamarla/heart-disease-prediction)  
- **Origin:** UCI Machine Learning Repository  
- **Target Variable:** `Heart Disease` (1 = Presence, 0 = Absence)

---

## 🔍 Project Description

The dataset contains patient health records including age, sex, chest pain type, cholesterol level, resting blood pressure, fasting blood sugar, ECG results, and more.  

We aim to build models that can identify patients at risk of heart disease using these features. The work done includes:

- Data cleaning and handling missing values
- Separating categorical and continuous variables
- Exploratory Data Analysis (EDA):
  - Count plots, bar plots, distribution plots
  - Correlation heatmap
- Data preprocessing:
  - Encoding categorical features
  - Feature scaling
- Train-test split
- Model training using multiple classification algorithms
- Performance comparison using accuracy, confusion matrix, and classification report

---

## 🧪 Models Trained

### ✅ Logistic Regression  
A statistical method that models the probability of a binary outcome using the logistic function.

### ✅ Naive Bayes Classifier  
A probabilistic model based on Bayes’ theorem, assuming independence between features. Fast and effective.

### ✅ Support Vector Classifier (SVC)  
Separates data using hyperplanes in a high-dimensional space, suitable for complex but small to medium-sized datasets.

### ✅ K-Nearest Neighbors (KNN)  
A non-parametric algorithm that classifies based on the majority class among the k-nearest data points.

### ✅ Decision Tree Classifier  
Builds a flowchart-like structure of decisions, useful for interpretability.

### ✅ Random Forest Classifier  
An ensemble of decision trees with better generalization, less overfitting, and higher accuracy.

### ✅ XGBoost Classifier  
A powerful gradient boosting technique optimized for performance and scalability.

---

## 📚 Libraries Used

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

---

## 📊 Model Accuracies

| Model                  | Accuracy       |
|------------------------|----------------|
| Logistic Regression    | 83.95%         |
| Naive Bayes            | 82.72%         |
| Support Vector         | 79.01%         |
| K Nearest Neighbours   | 83.95%         |
| Decision Tree          | 80.25%         |
| Random Forest          | 🏆 85.18%       |
| XGBoost                | 80.25%         |

> 🟢 **Random Forest Classifier** gave the best accuracy among all models.

---

## 📝 Conclusion

- Random Forest was the top performer in terms of accuracy, followed closely by KNN and Logistic Regression.
- This project strengthened my understanding of classification techniques, evaluation metrics, and model comparison.
- Given the serious global impact of heart disease, such models could assist healthcare providers in early detection and preventive action.

---

## 🚀 Future Improvements

- Deploy the best-performing model via a web app using **Streamlit** or **Flask**
- Apply **hyperparameter tuning** and **cross-validation**
- Handle class imbalance using **SMOTE** or **weighted loss functions**
- Integrate real-time data from smart devices or medical records

---

## 👤 Author

**Kanishka Chouhan**  
Aspiring Data Scientist  


---

> 📌 *Note: This project is for academic and learning purposes. The dataset is not suitable for clinical decision-making without further validation.*

