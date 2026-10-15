# Internee Performance Predictor & Interactive Web App

This project features a machine-learning regression pipeline and an interactive Streamlit web application to predict intern performance scores based on key training metrics.

## 📋 Project Overview

- **Target Variable**: `Performance_Score` (Float)
- **Features Used**:
  - `Completion_Time` (hours) - Time taken by interns to complete tasks.
  - `Feedback_Rating` (1 to 5 scale) - Ratings received from mentors or coordinators.
  - `Attendance` (percentage) - Percentage of days attended.

---

## 🛠️ Data Preprocessing & Cleaning

1. **Feature Dropping**: Dropped `Intern_ID` as it acts only as an identifier and contains no predictive signal.
2. **Handling Missing Values**:
   - `Completion_Time` & `Attendance` nulls were filled with their respective column **mean** values.
   - `Feedback_Rating` nulls were filled with the column **median** value.
3. **Feature Transformation**:
   - `Feedback_Rating` float values were rounded to discrete integers to reflect standard evaluation ratings.

---

## 🧪 Machine Learning Models & Evaluation

We trained and evaluated several regression algorithms using an **80/20 train-test split**:

| Model | Hyperparameters | Mean Squared Error (MSE) | Root Mean Squared Error (RMSE) | Mean Absolute Error (MAE) |
| :--- | :--- | :---: | :---: | :---: |
| **Random Forest Regressor** | default (n_estimators=200, max_depth=15) | 112.10 | 10.59 | 8.55 |
| **Random Forest (Tuned)** | n_estimators=500, max_depth=5 | 100.10 | 10.00 | 8.16 |
| **XGBoost Regressor** | n_estimators=80, max_depth=3, lr=0.1 | **96.75** | **9.84** | **8.02** |
| **K-Neighbors Regressor** | n_neighbors=15 | 109.42 | 10.46 | 8.57 |
| **Linear Regression** | default | 100.92 | 10.05 | 8.17 |

---

## 🚀 Streamlit Interactive Web Application

An interactive web dashboard is included to let you make live predictions and dynamically swap between all four trained models (**XGBoost**, **Tuned Random Forest**, **Linear Regression**, and **KNN Regressor**).

### How to Run the App

1. **Install Dependencies**:
   ```bash
   pip install streamlit xgboost scikit-learn pandas numpy
   npm install -g localtunnel
   ```

2. **Launch the Streamlit Server**:
   ```bash
   streamlit run app.py
   ```

3. **Expose with Localtunnel (for Google Colab environment)**:
   ```bash
   npx localtunnel --port 8501
   ```
   Use your public IP (displayed in the notebook run cell) as the password to unlock the secure localtunnel URL.
