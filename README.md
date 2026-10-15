# Internee Performance Predictor

This project aims to build a machine-learning regression model to predict intern performance scores based on key training metrics such as task completion rates (completion time), task feedback, and attendance rates.

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
   - Replacements were chosen based on the distribution of each feature:
     - `Completion_Time` & `Attendance` nulls were filled with their respective column **mean** values.
     - `Feedback_Rating` nulls were filled with the column **median** value.
3. **Feature Transformation**:
   - Mentors usually evaluate using discrete ratings. Thus, the float values in `Feedback_Rating` were rounded to integers using the round function.

---

## 🧪 Machine Learning Models & Evaluation

We trained and evaluated several regression algorithms using an **80/20 train-test split**. Here are the results of our experiments:

| Model | Hyperparameters | Mean Squared Error (MSE) | Root Mean Squared Error (RMSE) | Mean Absolute Error (MAE) |
| :--- | :--- | :---: | :---: | :---: |
| **Random Forest Regressor** | default (n_estimators=200, max_depth=15) | 112.10 | 10.59 | 8.55 |
| **Random Forest (Tuned)** | n_estimators=500, max_depth=5 | 100.10 | 10.00 | 8.16 |
| **XGBoost Regressor** | n_estimators=80, max_depth=3, lr=0.1 | **96.75** | **9.84** | **8.02** |
| **K-Neighbors Regressor** | n_neighbors=15 | 109.42 | 10.46 | 8.57 |
| **Linear Regression** | default | 100.92 | 10.05 | 8.17 |

### 🔑 Key Findings
- **XGBoost Regressor** yielded the best overall performance with the lowest prediction error (**MAE of 8.02**).
- Restraining tree depth (e.g., `max_depth=5` in Random Forest) significantly helped avoid overfitting given the small feature set (3 features).

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/internee-performance-predictor.git
   ```
2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
   ```
3. Run the Jupyter/Google Colab Notebook to see the training pipeline and evaluations step-by-step.
