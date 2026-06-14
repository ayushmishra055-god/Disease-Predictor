# 🏥 Interactive Disease Predictor App

An end-to-end Machine Learning web application that predicts potential medical conditions based on user-selected symptoms. Built with **Python**, **Streamlit**, and **scikit-learn**, this project showcases a clean user interface integrated with a robust predictive modeling backend.

🚀 **[Live Demo Link](https://disease-predictor-emhjvypcsnp8dkh7tut5ye.streamlit.app/)**

---

## 📊 Project Overview
This application serves as an educational tool demonstrating how healthcare datasets can be leveraged to assist in symptom analysis. Users can select multiple symptoms from an intuitive multi-select interface, and a trained machine learning model evaluates the inputs in real-time to output the most likely condition along with a statistical confidence level.

### **Key Features:**
* **Dynamic Multi-Select Dropdown:** Converts snake_case dataset features into beautifully formatted, readable text for a seamless user experience.
* **Robust ML Backend:** Utilizes a **Random Forest Classifier** trained to handle multi-class classification cleanly.
* **On-the-Fly Predictions:** Generates immediate diagnostic predictions and computes the exact probability confidence scores.
* **Streamlit Caching Optimization:** Implements `@st.cache_resource` to ensure the dataset is read and the model is trained only once, delivering lightning-fast load times for users.

---

## 🛠️ Tech Stack & Libraries
* **Frontend Interface:** Streamlit (UI components, layout controls, state management)
* **Data Processing:** Pandas (Data cleanup, structural feature formatting)
* **Machine Learning:** Scikit-learn (RandomForestClassifier, ensemble modeling)
* **Language:** Python 3

---

## 📁 Repository Structure
```text
├── app.py                 # Core Streamlit application & prediction logic
├── disease_data.csv       # Cleaned symptom-to-disease training dataset
└── requirements.txt       # Cloud deployment dependency configuration
