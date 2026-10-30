import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Prediction Portal",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for Clean Hero Section and Layout
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background: linear-gradient(rgba(15, 12, 41, 0.90), rgba(48, 43, 99, 0.90)), 
                    url('https://images.unsplash.com/photo-1497215728101-856f4ea42174?auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 30px 20px 20px 20px;
        margin-bottom: 20px;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #bfeef5;
        margin-bottom: 10px;
        text-shadow: 0 4px 15px rgba(191, 238, 245, 0.3);
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #d0e8f2;
        max-width: 650px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Input Labels */
    label {
        color: #bfeef5 !important;
        font-weight: 600 !important;
    }

    /* Submit Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #bfeef5, #3a7bd5);
        color: #0f0c29;
        border: none;
        padding: 14px 20px;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 15px rgba(191, 238, 245, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        background: linear-gradient(90deg, #ffffff, #bfeef5);
        box-shadow: 0 6px 20px rgba(191, 238, 245, 0.6);
    }

    /* Result Box */
    .result-box {
        background: rgba(191, 238, 245, 0.15);
        border: 1px solid #bfeef5;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        max-width: 900px;
        margin: 20px auto;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section (Navbar completely removed)
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Gradient Boosting Prediction Model</div>
    <div class="hero-subtitle">
        Empower your workplace analytics with machine learning predictions. Select the employee metrics below to estimate probability percentage in real time.
    </div>
</div>
""", unsafe_allow_html=True)

# Model Loading Function
@st.cache_resource
def load_model():
    with open("gradient.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load kartana error aala: {e}")
    st.stop()

# Main Form Section
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        dept_dict = {
            "HR": 1,
            "Engineering": 2,
            "Sales": 3,
            "Marketing": 4,
            "Finance": 5
        }
        dept_selected = st.selectbox("Department", options=list(dept_dict.keys()))
        department = dept_dict[dept_selected]

        experience_years = st.slider("Experience (Years)", min_value=0.0, max_value=40.0, value=2.0, step=0.5)

        edu_dict = {
            "High School": 1,
            "Bachelor's Degree": 2,
            "Master's Degree": 3,
            "PhD / Doctorate": 4
        }
        edu_selected = st.selectbox("Education Level", options=list(edu_dict.keys()))
        education_level = edu_dict[edu_selected]

    with col2:
        age = st.slider("Age", min_value=18, max_value=70, value=28)

        gender_option = st.selectbox("Gender", options=["Female", "Male", "Other"])
        gender = 0 if gender_option == "Female" else (1 if gender_option == "Male" else 2)

        city_dict = {
            "Pune": 1,
            "Mumbai": 2,
            "Bangalore": 3,
            "Hyderabad": 4,
            "Delhi": 5
        }
        city_selected = st.selectbox("City", options=list(city_dict.keys()))
        city = city_dict[city_selected]

    submit_button = st.form_submit_button("🚀 Generate Prediction")

# Output Display Logic
if submit_button:
    input_data = np.array([[department, experience_years, education_level, age, gender, city]])
    
    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            if len(probabilities) == 2:
                pred_percentage = probabilities[1] * 100
            else:
                pred_percentage = np.max(probabilities) * 100
        else:
            prediction = model.predict(input_data)[0]
            pred_percentage = float(prediction) * 100 if prediction <= 1 else float(prediction)

        st.markdown(f"""
        <div class="result-box">
            <h3 style="color: #bfeef5; margin: 0;">Predicted Probability Result</h3>
            <h1 style="color: #ffffff; font-size: 3rem; margin: 10px 0;">{pred_percentage:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    except Exception as err:
        st.error(f"Prediction kartaana adchan aali: {err}")
