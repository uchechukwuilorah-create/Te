import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and feature names
@st.cache_resource
def load_model():
    model = joblib.load('student_model_1.pkl')
    features = joblib.load('feature_names_1.pkl')
    return model, features

model, features = load_model()

st.title("🎓 Student Performance Predictor")
st.markdown("""
This app predicts the **Grade Class** of a student based on various performance and demographic factors.
The Grade Class ranges from 0 to 4, where 0 represents the highest performance and 4 represents the lowest.
""")

st.sidebar.header("Input Student Data")

def user_input_features():
    age = st.sidebar.slider("Age", 15, 18, 16)
    gender = st.sidebar.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    ethnicity = st.sidebar.selectbox("Ethnicity", options=[0, 1, 2, 3], format_func=lambda x: ["Caucasian", "African American", "Asian", "Other"][x])
    parental_edu = st.sidebar.selectbox("Parental Education", options=[0, 1, 2, 3, 4], format_func=lambda x: ["None", "High School", "Some College", "Bachelor's", "Higher"][x])
    study_time = st.sidebar.slider("Weekly Study Time (hours)", 0.0, 20.0, 10.0)
    absences = st.sidebar.slider("Absences", 0, 30, 5)
    tutoring = st.sidebar.selectbox("Tutoring", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    parental_support = st.sidebar.selectbox("Parental Support", options=[0, 1, 2, 3, 4], format_func=lambda x: ["None", "Low", "Moderate", "High", "Very High"][x])
    extracurricular = st.sidebar.selectbox("Extracurricular Activities", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    sports = st.sidebar.selectbox("Sports", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    music = st.sidebar.selectbox("Music", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    volunteering = st.sidebar.selectbox("Volunteering", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    gpa = st.sidebar.slider("GPA", 0.0, 4.0, 2.5)
    
    data = {
        'Age': age,
        'Gender': gender,
        'Ethnicity': ethnicity,
        'ParentalEducation': parental_edu,
        'StudyTimeWeekly': study_time,
        'Absences': absences,
        'Tutoring': tutoring,
        'ParentalSupport': parental_support,
        'Extracurricular': extracurricular,
        'Sports': sports,
        'Music': music,
        'Volunteering': volunteering,
        'GPA': gpa
    }
    features_df = pd.DataFrame(data, index=[0])
    return features_df

input_df = user_input_features()

st.subheader("Student Features")
st.write(input_df)

if st.button("Predict Grade Class"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    grade_map = {
        0: "A (Excellent)",
        1: "B (Good)",
        2: "C (Average)",
        3: "D (Below Average)",
        4: "F (Fail)"
    }
    
    st.subheader("Prediction")
    st.success(f"The predicted Grade Class is: **{grade_map[int(prediction[0])]}**")
    
    st.subheader("Prediction Probability")
    prob_df = pd.DataFrame(prediction_proba, columns=[grade_map[i] for i in range(5)])
    st.bar_chart(prob_df.T)

st.markdown("---")
st.info("Note: This model was trained on the provided student performance dataset using a Random Forest Classifier.")
