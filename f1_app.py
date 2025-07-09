import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="🏁 F1 Race Winner Predictor", layout="centered")

@st.cache_data
def load_data():
    df = pd.read_csv("race_data_full.csv")

    # Check required columns
    required = ['grid', 'constructor', 'circuit', 'winner']
    if not all(col in df.columns for col in required):
        raise ValueError(f"CSV must contain columns: {required}")

    # Encode categorical columns
    mapper = {}
    for col in ['constructor', 'circuit']:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        mapper[col] = le

    # Features and target
    X = df[['grid', 'constructor_encoded', 'circuit_encoded']]
    y = df['winner']
    return X, y, df, mapper

# Load and train
try:
    X, y, df, mapper = load_data()
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    accuracy = model.score(X, y)

    # UI
    st.title("🏁 F1 Race Winner Predictor")
    st.markdown("Predict if a constructor will win based on starting grid, circuit, and team.")

    st.sidebar.header("🔧 Input Configuration")

    constructor = st.sidebar.selectbox("Select Constructor", mapper['constructor'].classes_)
    circuit = st.sidebar.selectbox("Select Circuit", mapper['circuit'].classes_)
    grid = st.sidebar.slider("Starting Grid Position", 1, 20, 5)

    # Encode user input
    constructor_enc = mapper['constructor'].transform([constructor])[0]
    circuit_enc = mapper['circuit'].transform([circuit])[0]
    user_input = [[grid, constructor_enc, circuit_enc]]

    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0][1]

    st.subheader("✅ Model Accuracy")
    st.metric("Training Accuracy", f"{accuracy:.2f}")

    st.subheader("🔮 Prediction")
    st.write(f"**Constructor:** {constructor} | **Circuit:** {circuit} | **Grid:** P{grid}")

    st.progress(probability, text="Predicted Win Probability")
    if prediction == 1:
        st.success(f"🎉 {constructor} is predicted to **WIN** from P{grid} at {circuit}!")
    else:
        st.error(f"❌ {constructor} is predicted to **NOT win** from P{grid} at {circuit}.")

    # Past data
    st.subheader("📊 Historical Win Rates")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Constructor Win Rate", f"{100 * df[df['constructor'] == constructor]['winner'].mean():.1f}%")
    with col2:
        st.metric("Circuit Win Rate", f"{100 * df[df['circuit'] == circuit]['winner'].mean():.1f}%")

except Exception as e:
    st.error(f"❌ App failed to load. Error: {e}")
