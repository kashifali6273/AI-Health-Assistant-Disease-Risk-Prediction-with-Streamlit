import os
import joblib
import streamlit as st
import numpy as np

# =======================
# Setup Paths
# =======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIABETES_MODEL_PATH = os.path.join(BASE_DIR, "models", "diabetes_model.pkl")
HEART_MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_model.pkl")

# Load models
diabetes_model = joblib.load(DIABETES_MODEL_PATH)
heart_model = joblib.load(HEART_MODEL_PATH)

# Inspect classes_
diabetes_classes = getattr(diabetes_model, "classes_", np.array([0, 1]))
heart_classes = getattr(heart_model, "classes_", np.array([0, 1]))

# Prototype examples used to infer which class corresponds to disease
def _prototypes(dataset: str):
    if dataset == "diabetes":
        low = np.array([[1, 85, 66, 29, 0, 26.6, 0.35, 25]])
        high = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
    else:  # heart
        # low-risk prototype
        low = np.array([[40, 0, 0, 120, 200, 0, 0, 170, 0, 0.0, 1, 0]])
        # high-risk prototype
        high = np.array([[58, 1, 2, 150, 280, 1, 2, 120, 1, 2.5, 2, 2]])
    return low, high

def determine_disease_label(model, classes, dataset_name):
    """
    Heuristic: compare model.predict_proba on a high-risk vs low-risk prototype.
    Choose the class index whose probability increases the most for the high-risk example.
    This gives the label representing 'disease' for that model.
    """
    if not hasattr(model, "predict_proba"):
        # fallback: assume label '1' means disease if present
        return 1 if 1 in classes else classes[0]

    try:
        low, high = _prototypes(dataset_name)
        p_low = model.predict_proba(low)[0]
        p_high = model.predict_proba(high)[0]
        diffs = p_high - p_low
        idx = int(np.argmax(diffs))
        disease_label = classes[idx]
        return disease_label
    except Exception:
        return 1 if 1 in classes else classes[0]

# Determine labels at startup
diabetes_disease_label = determine_disease_label(diabetes_model, list(diabetes_classes), "diabetes")
heart_disease_label = determine_disease_label(heart_model, list(heart_classes), "heart")

def display_result(probs, classes, disease_label, disease_name):
    """
    Show percentage bar and human-friendly message.
    probs: array-like aligned with classes
    disease_label: label (e.g. 1) corresponding to disease presence
    """
    # find index of disease_label in classes
    try:
        disease_idx = list(classes).index(disease_label)
    except ValueError:
        disease_idx = 1 if 1 in classes else 0

    disease_prob = float(probs[disease_idx])
    pct = max(0, min(100, int(disease_prob * 100)))

    # show numeric and progress bar
    st.write(f"### Risk Score: **{disease_prob*100:.1f}%**")
    st.progress(pct)

    # message based on threshold
    if disease_prob >= 0.5:
        st.error(f"❌ High Risk of {disease_name} — {disease_prob*100:.1f}%")
    else:
        st.success(f"✅ Low Risk of {disease_name} — {disease_prob*100:.1f}%")

    # additionally show both class probabilities in a small table for transparency
    prob_map = {str(c): f"{p*100:.1f}%" for c, p in zip(classes, probs)}
    st.write("Class probabilities:", prob_map)

# =======================
# UI Config
# =======================
st.set_page_config(page_title="AI Health Assistant", page_icon="🩺", layout="wide")
st.title("🩺 AI Health Assistant")
st.write("A modern health prediction tool for **Diabetes** and **Heart Disease**.")

# Sidebar
choice = st.sidebar.selectbox("Choose Prediction Type", ["Diabetes Prediction", "Heart Disease Prediction"])

# =======================
# Diabetes Prediction
# =======================
if choice == "Diabetes Prediction":
    st.header("🧬 Diabetes Risk Prediction")

    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1, value=85)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1, value=66)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1, value=29)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1, value=0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1, value=26.6)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01, value=0.35)
    age = st.number_input("Age", min_value=1, max_value=120, step=1, value=25)

    if st.button("🔍 Predict Diabetes"):
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                              insulin, bmi, dpf, age]])
        # get prediction/prob
        prediction = diabetes_model.predict(features)[0]
        probs = diabetes_model.predict_proba(features)[0]
        # display using detected disease label
        display_result(probs, list(diabetes_classes), diabetes_disease_label, "Diabetes")

# =======================
# Heart Disease Prediction
# =======================
elif choice == "Heart Disease Prediction":
    st.header("❤️ Heart Disease Risk Prediction")

    age = st.number_input("Age", min_value=1, max_value=120, step=1, value=40)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1], index=0)
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], index=0)
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=250, step=1, value=120)
    chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, step=1, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1], index=0)
    restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2], index=0)
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=60, max_value=250, step=1, value=170)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1], index=0)
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1, value=0.0)
    slope = st.selectbox("Slope of Peak Exercise ST (slope)", [0, 1, 2], index=1)
    ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4], index=0)

    if st.button("🔍 Predict Heart Disease"):
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca]])
        prediction = heart_model.predict(features)[0]
        probs = heart_model.predict_proba(features)[0]
        display_result(probs, list(heart_classes), heart_disease_label, "Heart Disease")
