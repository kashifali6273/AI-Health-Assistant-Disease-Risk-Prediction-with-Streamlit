# 🩺 AI Health Assistant

This is a simple **AI-powered health prediction tool** built using **Streamlit**.  
Right now, it can predict the risk of **Diabetes** and **Heart Disease** using machine learning models.  

The project is designed to be easy to use, modern in design, and extendable for future features.

---

## 🚀 Features
- **Diabetes Prediction** 🧬  
  Enter details like glucose level, BMI, insulin, etc., and get prediction with probability.  

- **Heart Disease Prediction** ❤️  
  Enter health indicators like blood pressure, cholesterol, chest pain type, etc., and get risk assessment.  

- **Probability Bar** 📊  
  Predictions are shown with a clear percentage bar for confidence level.  

- **Modern Streamlit UI** 🎨  
  Simple, clean, and responsive web interface.

---

## 🛠️ Tech Stack
- **Python**
- **Streamlit** – for frontend interface
- **Scikit-learn** – for machine learning models
- **Joblib** – to save/load models
- **NumPy** – for numerical processing

---

## 📂 Project Structure
ai-health-assistant/
│
├── app/
│ └── streamlit_app.py # Main Streamlit UI
│
├── models/
│ ├── diabetes_model.pkl # Trained Diabetes model
│ └── heart_model.pkl # Trained Heart Disease model
│
├── services/
│ └── api.py # API file (future integration)
│
├── venv/ # Virtual environment
└── README.md

---

## ▶️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/ai-health-assistant.git
   cd ai-health-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run app/streamlit_app.py
   ```

---

## ✅ Example Predictions

**Diabetes Prediction:**  
- Glucose = 150, BMI = 35 → High Risk  
- Glucose = 100, BMI = 22 → Low Risk  

**Heart Disease Prediction:**  
- Age = 60, Cholesterol = 280, High Blood Pressure → High Risk  
- Age = 25, Cholesterol = 180, Normal BP → Low Risk  

---

## 🔮 Future Improvements
This project can be extended with more advanced features:

- 🤖 RAG-based Chatbot (explain predictions and give advice)  
- 📈 More Diseases (Cancer prediction, Kidney disease, etc.)  
- 🌍 Dashboard for health monitoring  
- 📱 Mobile App version  
- ☁️ Deploy on Cloud (Streamlit Cloud, Hugging Face Spaces, or Heroku)  

---

## 📌 Notes
This tool is for educational purposes only and not a substitute for professional medical advice.  

Always consult a doctor for health concerns.

---

## 👨‍💻 Author
Developed by **Kashif Ali**  

For learning AI, ML, and health applications.
