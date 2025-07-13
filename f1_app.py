import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ✅ Page setup
st.set_page_config(
    page_title="🏁 F1 Predictor",
    page_icon="🏎️",
    layout="centered"
)

# ✅ Sidebar info
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
This app predicts Formula 1 race outcomes using a Random Forest machine learning model.

You can:
- Select a **constructor**
- Choose a **circuit**
- Pick a **driver**
- Set a **starting grid position**

It will calculate the probability of winning the race based on historical data.

---
**Made with ❤️ by Pravee**
""")

# ✅ Load data
@st.cache_data
def load_data():
    df = pd.read_csv("race_data_full.csv")
    df.columns = [col.lower() for col in df.columns]

    for col in ['grid', 'constructor', 'circuit', 'driver', 'winner']:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    mapper, df_enc = {}, df.copy()
    for col in ['constructor', 'circuit', 'driver']:
        le = LabelEncoder()
        df_enc[col + '_enc'] = le.fit_transform(df[col])
        mapper[col] = le

    return df_enc, mapper

# ✅ Load and train
df, mapper = load_data()
X = df[['grid','constructor_enc','circuit_enc','driver_enc']]
y = df['winner']
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X, y)

# ✅ Main app UI
st.title("🏁 F1 Race Winner Predictor")

constructor = st.selectbox("🏎️ Choose Constructor", sorted(df['constructor'].unique()))
circuit = st.selectbox("📍 Choose Circuit", sorted(df['circuit'].unique()))
driver = st.selectbox("👤 Choose Driver", sorted(df['driver'].unique()))
grid = st.slider("🎯 Grid Position (1 = Pole)", 1, 20, 5)

# ✅ Encode input
enc = {
    "constructor": mapper['constructor'].transform([constructor])[0],
    "circuit": mapper['circuit'].transform([circuit])[0],
    "driver": mapper['driver'].transform([driver])[0]
}
X_input = [[grid, enc['constructor'], enc['circuit'], enc['driver']]]
pred = model.predict(X_input)[0]
prob = model.predict_proba(X_input)[0][1]

# ✅ Prediction result
st.subheader("🔮 Prediction")
st.write(f"**{driver}** ({constructor}) from P{grid} at **{circuit}**")

st.progress(prob, text=f"Win probability: {prob:.2%}")
if pred:
    st.success("🎉 Predicted to WIN!")
else:
    st.error("❌ Predicted to NOT win.")

# ✅ Metrics
st.subheader("📊 Historical Win Rates")
c1, c2, c3 = st.columns(3)
c1.metric("Driver Win %", f"{100 * df[df['driver'] == driver]['winner'].mean():.1f}%")
c2.metric("Constructor Win %", f"{100 * df[df['constructor'] == constructor]['winner'].mean():.1f}%")
c3.metric("Circuit Win %", f"{100 * df[df['circuit'] == circuit]['winner'].mean():.1f}%")

# ✅ Win rate vs grid
st.subheader("📈 Win Rate vs Grid Position")
grid_win = df.groupby('grid')['winner'].mean() * 100
fig, ax = plt.subplots()
ax.plot(grid_win.index, grid_win.values, marker='o')
ax.set_xlabel('Grid Position')
ax.set_ylabel('Win Rate (%)')
ax.set_title('🏁 Grid Position vs Win Rate')
st.pyplot(fig)
