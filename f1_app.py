import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data and cache it for performance
@st.cache_data
def load_data():
    df = pd.read_csv("race_data_full.csv")

    # Encode categorical columns
    mapper = {
        'constructor': LabelEncoder(),
        'circuit': LabelEncoder()
    }

    for col, le in mapper.items():
        df[col + '_encoded'] = le.fit_transform(df[col])

    # Features and label
    features = ['grid', 'constructor_encoded', 'circuit_encoded']
    X = df[features]
    y = df['winner']

    return X, y, df, mapper

# Train model
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# App UI
st.title("🏁 F1 Race Winner Predictor")
st.markdown("### Predict if a constructor will win the race based on grid position, circuit, and constructor name.")

# Load and train
X, y, df, mapper = load_data()
model, accuracy = train_model(X, y)

# Show model accuracy
st.success(f"✅ Model Accuracy: {accuracy:.2f}")

# User inputs
constructor_list = sorted(df['constructor'].unique())
circuit_list = sorted(df['circuit'].unique())

selected_constructor = st.selectbox("🏎 Select Constructor", constructor_list)
selected_circuit = st.selectbox("📍 Select Circuit", circuit_list)
grid_position = st.number_input("🔢 Starting Grid Position (1 = Pole)", min_value=1, max_value=20, value=1)

# Prediction
if st.button("🔮 Predict Winner"):
    input_data = pd.DataFrame({
        'grid': [grid_position],
        'constructor_encoded': [mapper['constructor'].transform([selected_constructor])[0]],
        'circuit_encoded': [mapper['circuit'].transform([selected_circuit])[0]]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success(f"✅ {selected_constructor} starting at P{grid_position} is predicted to WIN at {selected_circuit}!")
    else:
        st.error(f"❌ {selected_constructor} starting at P{grid_position} is predicted to NOT win at {selected_circuit}.")
