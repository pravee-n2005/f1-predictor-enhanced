import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load race data
df = pd.read_csv("race_data.csv")

# 2. Add winner column
df["winner"] = (df["position"] == 1).astype(int)

# 3. Encode constructor
le = LabelEncoder()
df["constructor_encoded"] = le.fit_transform(df["constructor"])

# 4. Prepare features and labels
X = df[["grid", "constructor_encoded"]]
y = df["winner"]

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 7. Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2f}")

# 8. Visualize feature importance
sns.barplot(x=model.feature_importances_, y=["grid", "constructor_encoded"])
plt.title("Feature Importance")
plt.show()

# 9. Get user input
team_input = input("\nEnter team name (e.g. Red Bull): ").strip()
grid_input = input("Enter grid position (e.g. 2): ").strip()

try:
    grid_input = int(grid_input)
except ValueError:
    print("❌ Invalid grid position. Must be a number.")
    exit()

# 10. Predict win
if team_input in le.classes_:
    team_encoded = le.transform([team_input])[0]
    input_data = pd.DataFrame(
        np.array([[grid_input, team_encoded]]),
        columns=["grid", "constructor_encoded"]
    )
    prediction = model.predict(input_data)

    print("\n🔮 Prediction:")
    if prediction[0] == 1:
        print(f"✅ {team_input} starting at P{grid_input} is predicted to WIN!")
    else:
        print(f"❌ {team_input} starting at P{grid_input} is predicted to NOT win.")
else:
    print(f"❌ Team '{team_input}' not found in training data.")
