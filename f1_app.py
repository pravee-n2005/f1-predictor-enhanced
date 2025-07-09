import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def load_data():
    df = pd.read_csv("race_data_full.csv")

    # Create team and track strength features
    for col in ['constructor', 'raceName']:
        df[col + '_strength'] = df.groupby(col)['winner'].transform('mean')

    # Encode categorical features
    for col in ['constructor', 'raceName']:
        le = LabelEncoder()  # ✅ define it here
        df[col + '_encoded'] = le.fit_transform(df[col])

    return df
