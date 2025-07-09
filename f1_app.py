import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import random

@st.cache_data
def load_data():
    df = pd.read_csv("race_data_full.csv")
    df["grid_bin"] = pd.cut(df["grid"], bins=[0,5,10,20], labels=["Front","Mid","Back"])
    for col in ['constructor', 'driver', 'raceName', 'grid_bin']:
        df[col + '_strength'] = df.groupby(col)['winner'].transform('mean')
    return df.dropna()

df = load_data()

# Encode
mapper = {col: LabelEncoder().fit for col in ['constructor','driver','raceName','grid_bin']}
for col, le in mapper.items():
    df[col + '_encoded'] = le.fit_transform(df[col])

# Features
features = ['grid','constructor_encoded','driver_encoded','raceName_encoded',
            'grid_bin_encoded','constructor_strength','driver_strength','raceName_strength']
X = df[features]
y = df['winner']

# Train
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# UI here (adapt from previous version, using model.predict & predict_proba)
