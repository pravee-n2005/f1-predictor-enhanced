import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# --------------------
# 🚀 App Configuration
# --------------------
st.set_page_config(page_title="🏁 F1 Race Predictor", layout="centered")

# ----------------------
# 📦 Load & Preprocess
# ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("race_data_full.csv")

    required_cols = ['grid', 'constructor', 'circuit', 'winner']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    mapper = {}
    for col in ['constructor', 'circuit']:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        mapper[col] = le

    X = df[['grid', 'constructor_encoded', 'circuit_encoded']]
    y = df['winner']
    return X, y, df, mapper

# -------------------------
# 🧠 Train ML Model
# -------------------------
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    return model

# -------------------------
# 🎯 Make Prediction
# -------------------------
def predict_win(model, mapper, grid, constructor, circuit):
    constructor_encoded = mapper['constructor'].transform([constructor])[0]
    circuit_encoded = mapper['circuit'].transform([circuit])[0]
    user_input = [[grid, constructor_encoded, circuit_encoded]]
    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0][1]
    return prediction, probability

# -------------------------
# 📊 Win Rate by Grid
# -------------------------
def plot_win_rate(df):
    grid_win = df.groupby('grid')['winner'].mean() * 100
    fig, ax = plt.subplots()
    ax.plot(grid_win.index, grid_win.values, marker='o')
    ax.set_xlabel('Starting Grid Position')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('📈 Win Rate vs Grid Position')
    ax.grid(True)
    return fig

# -------------------------
# 🖥️ UI
# -------------------------
try:
    X, y, df, mapper = load_data()
    model = train_model(X, y)
    accuracy = model.score(X, y)

    st.title("🏁 F1 Race Winner Predictor")
    st.markdown("Predict the chances of a constructor winning a race based on grid, constructor and track.")

    st.sidebar.header("🧮 Input Configuration")
    constructor = st.sidebar.selectbox("Select Constructor", mapper['constructor'].classes_)
    circuit = st.sidebar.selectbox("Select Circuit", mapper['circuit'].classes_)
    grid = st.sidebar.slider("Starting Grid Position", 1, 20, 5)

    prediction, probability = predict_win(model, mapper, grid, constructor, circuit)

    # --- Model Accuracy
    st.subheader("✅ Model Accuracy")
    st.metric("Training Accuracy", f"{accuracy:.2f}")

    # --- Prediction Result
    st.subheader("🔮 Prediction")
    st.write(f"**Constructor:** {constructor} | **Circuit:** {circuit} | **Grid:** P{grid}")
    st.progress(probability, text=f"Predicted Win Probability: {probability:.2%}")

    if prediction == 1:
        st.success(f"🎉 {constructor} is predicted to WIN from P{grid} at {circuit}!")
    else:
        st.error(f"❌ {constructor} is predicted to NOT win from P{grid} at {circuit}.")

    # --- Past Win Rates
    st.subheader("📊 Historical Performance")

    col1, col2 = st.columns(2)
    with col1:
        rate = df[df['constructor'] == constructor]['winner'].mean()
        st.metric("Constructor Win Rate", f"{100 * rate:.1f}%")
    with col2:
        rate = df[df['circuit'] == circuit]['winner'].mean()
        st.metric("Circuit Win Rate", f"{100 * rate:.1f}%")

    # --- Grid win rate chart
    st.subheader("📈 Grid Position Impact")
    st.pyplot(plot_win_rate(df))

except Exception as e:
    st.error(f"🚨 App failed to load.\n\nDetails:\n{e}")
