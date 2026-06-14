import pandas as pd
import sklearn
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
st.set_page_config(page_title="Disease Predictor", page_icon="🏥", layout="centered")
st.title("🏥 Disease Predictor")
st.write("Select the symptoms you are experiencing given below and our ML model will predict the potential condition")
st.markdown("---")

@st.cache_resource
def train_model():
  
    df = pd.read_csv("disease_data.csv")

    # Remove leading/trailing spaces from column headers
    df.columns = df.columns.str.strip()

    # 2.Find the target column safely, even if case is slightly off
    target_column = None
    for col in df.columns:
        if col.lower() in ("disease", "prognosis"):
            target_column = col
            break

    # If we still can't find it, raise a helpful error message on screen
    if not target_column:
        st.error(
            f"Could not find a 'Disease' or 'prognosis' column. Found columns: {list(df.columns)}"
        )
        st.stop()

    # Features (X) are all symptom columns, Target (y) is the disease column
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Get a clean list of all unique symptom features
    symptom_features = list(X.columns)

    # Train a robust Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, symptom_features

with st.spinner("Please wait!! Your app is getting ready"):
  model, all_symptoms=train_model()

st.subheader("🕵️‍♂️ Select the symptoms")
readable_symptoms=[s.replace("_"," ").title() for s in all_symptoms]
selected_readable=st.multiselect(
  "What symptoms you are feeling(you can select multiple options)", options=readable_symptoms
)
st.markdown("---")

#prediction logic
st.subheader("📊 Diagnostic Prediction")
if st.button("Predict Condition",type="primary"):
  if len(selected_readable) ==0:
    st.warning("⚠️ Please select at least one symptom before predicting.")
  else:
    selected_converted=[s.replace(" ","_").lower() for s in selected_readable]
    #exact fromat
    input_data={symptom: 0 for symptom in all_symptoms}
    for symptom in selected_converted:
      if symptom in input_data:
        input_data[symptom]=1
    
    input_df=pd.DataFrame([input_data])
    #prediction
    prediction=model.predict(input_df)[0]
    probabilities=model.predict_proba(input_df)[0]
    max_proba=max(probabilities) *100
    st.success(f"### Predicted Condition: **{prediction}**")
    st.info(f"📈 **Confidence level:** {max_proba:.2f}%")

    st.markdown(" ⚠️ **Disclaimer:** This tool is for educational purposes only and powered by basic machine learning. It does not replace professional medical advice, diagnosis, or treatment.")