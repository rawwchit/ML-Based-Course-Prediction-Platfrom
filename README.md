# Course-Prediction-Platfrom
# 🎓 Student Performance Predictor using Machine Learning

## 📌 Project Overview

The Student Performance Predictor is a Machine Learning project that predicts a student's **Math Score** based on various academic and demographic factors such as:

- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course
- Reading Score
- Writing Score

The project uses **Linear Regression** to analyze relationships between these factors and predict student performance.

A **Streamlit Web Application** is also developed to provide an interactive interface for users to enter student details and obtain predictions instantly.

---

## 🚀 Features

- Data preprocessing using Label Encoding
- Machine Learning model training using Linear Regression
- Model evaluation using R² Score
- Interactive Streamlit web application
- Real-time student score prediction
- User-friendly interface

---

## 📂 Dataset

Dataset Used:

**Students Performance in Exams Dataset**

Dataset Features:

| Feature | Description |
|----------|------------|
| Gender | Student Gender |
| Race/Ethnicity | Student Group |
| Parental Level of Education | Parent Education Qualification |
| Lunch | Standard or Free/Reduced Lunch |
| Test Preparation Course | Completed or Not Completed |
| Reading Score | Reading Exam Marks |
| Writing Score | Writing Exam Marks |
| Math Score | Target Variable |

Dataset Size:

- 1000 Records
- 8 Columns

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Jupyter Notebook
- Streamlit

---

## ⚙️ Machine Learning Workflow


Dataset
   ↓
Data Preprocessing
   ↓
Label Encoding
   ↓
Train-Test Split
   ↓
Linear Regression Model
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Streamlit Deployment
____

## 🔄 Data Preprocessing

Machine Learning models cannot process text values directly.

Therefore, categorical features were converted into numerical values using Label Encoding.

Encoded Features:

* Gender
* Race/Ethnicity
* Parental Level of Education
* Lunch
* Test Preparation Course

⸻

## 🤖 Model Used

Linear Regression

Linear Regression is a supervised machine learning algorithm used to predict continuous numerical values.

The model was trained using:

* 80% Training Data
* 20% Testing Data

⸻

## 📈 Model Performance

Evaluation Metric:

R² Score
Accuracy: 0.8838

Interpretation

The model explains approximately 88.38% of the variation in student math scores.

This indicates strong predictive performance.
____

## 📊 Sample Prediction
Gender: Male
Race/Ethnicity: Group C
Parental Education: Bachelor's Degree
Lunch: Standard
Test Preparation: Completed
Reading Score: 80
Writing Score: 85
____

## 📁 Project Structure
Student_Performance_Predictor/
│
├── StudentsPerformance.csv
├── student_performance_predictor.ipynb
├── app.py
├── README.md
└── ml_project/
____

## 🎯 Advantages

* Predicts student performance efficiently
* Easy-to-use interface
* Real-world educational analytics application
* Demonstrates machine learning concepts clearly
* Can be extended with advanced ML algorithms
____

## ⚠️ Limitations

* Limited dataset size
* Uses only Linear Regression
* Does not consider external academic factors
____

## 🔮 Future Enhancements

* Random Forest Regression
* XGBoost Regression
* Feature Importance Visualization
* Interactive Dashboards
* Cloud Deployment
* Database Integration
____

## 🏆 Conclusion

This project successfully demonstrates the application of Machine Learning in educational analytics.

Using Linear Regression, the model achieved an accuracy of 88.38% and was integrated into a Streamlit web application for real-time predictions.

The project highlights how machine learning can assist educators and students in understanding academic performance trends.
```text
