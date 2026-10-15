import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Page layout
st.set_page_config(page_title="Internee Performance Predictor", layout="centered")

st.title("🎓 Internee Performance Predictor")
st.write("Predict intern performance scores based on key training metrics.")

# Load and Cache Data
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("intern_dataset_realistic (1).csv")
    df.drop(columns=["Intern_ID"], inplace=True, errors='ignore')
    
    # Preprocessing
    df['Feedback_Rating'] = np.round(df['Feedback_Rating'])
    
    # Imputations
    df['Completion_Time'] = df['Completion_Time'].fillna(df['Completion_Time'].mean())
    df['Attendance'] = df['Attendance'].fillna(df['Attendance'].mean())
    df['Feedback_Rating'] = df['Feedback_Rating'].fillna(df['Feedback_Rating'].median())
    
    return df

try:
    df = load_and_clean_data()
    
    # Re-train models locally for prediction
    X = df.drop(columns='Performance_Score')
    y = df['Performance_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.8)
    
    @st.cache_resource
    def train_models():
        models = {
            "XGBoost Regressor": XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.1, random_state=42).fit(X_train, y_train),
            "Random Forest (Tuned)": RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42).fit(X_train, y_train),
            "Linear Regression": LinearRegression().fit(X_train, y_train),
            "K-Neighbors Regressor": KNeighborsRegressor(n_neighbors=15).fit(X_train, y_train)
        }
        return models

    models = train_models()

    # User Inputs
    st.sidebar.header("⚙️ Input Parameters")
    completion_time = st.sidebar.slider("Completion Time (hours)", 1.0, 7.0, float(df['Completion_Time'].mean()), 0.01)
    feedback_rating = st.sidebar.selectbox("Feedback Rating (1 to 5)", [1, 2, 3, 4, 5], index=2)
    attendance = st.sidebar.slider("Attendance (%)", 0.0, 100.0, float(df['Attendance'].mean()), 0.1)
    
    selected_model_name = st.selectbox("🤖 Select Machine Learning Model", list(models.keys()))
    
    # Predict button
    if st.button("🔮 Predict Performance Score"):
        model = models[selected_model_name]
        input_data = pd.DataFrame([[
            completion_time,
            feedback_rating,
            attendance
        ]], columns=['Completion_Time', 'Feedback_Rating', 'Attendance'])
        
        prediction = model.predict(input_data)[0]
        
        st.success(f"✨ Predicted Performance Score: **{prediction:.2f}**")
        
        # Contextual metrics display
        st.metric(label="Predicted Score", value=f"{prediction:.2f} / 100")
        
except Exception as e:
    st.error(f"Error initializing application: {e}")
