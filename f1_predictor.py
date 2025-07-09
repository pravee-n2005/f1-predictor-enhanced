import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv("race_data_full.csv")

# Encode categorical features
mapper = {}
for col in ['constructor', 'circuit', 'driver']:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    mapper[col] = le

# Features and target
X = df[['grid', 'constructor_encoded', 'circuit_encoded', 'driver_encoded']]
y = df['winner']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# Save model and encoders
joblib.dump(model, "model.pkl")
joblib.dump(mapper, "label_encoders.pkl")
