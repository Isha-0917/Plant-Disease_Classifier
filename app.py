import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# === Load Trained Model and Encoder ===
model = joblib.load("model.joblib")      # XGBoost model
encoder = joblib.load("encoder.joblib")  # LabelEncoder for target

# === Load Dataset ===
df = pd.read_csv("plant_disease_classifier_sample.csv")

# === Streamlit Page Config ===
st.set_page_config(page_title="🌿 Plant Disease Classifier", layout="centered")

st.title("🌱 Plant Disease Classifier App")
st.markdown("Predict plant diseases based on leaf features using a trained XGBoost model. Built with 💚 Streamlit.")

# === Sidebar ===
st.sidebar.header("🔎 About")
st.sidebar.markdown("""
This app uses a machine learning model trained on plant disease data.
Enter features to predict the plant disease. You can also preview the dataset.
""")

# === View Dataset Toggle ===
if st.checkbox("📊 View Sample Dataset"):
    st.dataframe(df.head())

# === User Input Form ===
st.subheader("🔬 Enter Leaf Features:")
form = st.form(key="prediction_form")

input_data = {}
feature_columns = [col for col in df.columns if col != "Disease"]

for feature in feature_columns:
    dtype = df[feature].dtype
    if np.issubdtype(dtype, np.number):
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        default = float(df[feature].mean())
        input_data[feature] = form.slider(f"{feature}", min_value=min_val, max_value=max_val, value=default)
    else:
        options = df[feature].unique().tolist()
        input_data[feature] = form.selectbox(f"{feature}", options)

submitted = form.form_submit_button("🔍 Predict Disease")

# === Prediction ===
if submitted:
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    for col in input_df.select_dtypes(include=['object']).columns:
        input_df[col] = LabelEncoder().fit(df[col]).transform(input_df[col])  # or load saved encoders if needed

    prediction = model.predict(input_df)[0]
    predicted_label = encoder.inverse_transform([prediction])[0]
    st.success(f"🌾 Predicted Disease: **{predicted_label}**")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)
        prob_df = pd.DataFrame(proba, columns=encoder.inverse_transform(np.arange(len(encoder.classes_))))
        st.write("📈 Prediction Probabilities:")
        st.dataframe(prob_df.T.rename(columns={0: "Probability"}))
