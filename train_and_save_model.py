# train_and_save_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Load dataset
df = pd.read_csv("insurance.csv")

# Preprocess dataset
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df['region'] = df['region'].map({'southwest': 1, 'southeast': 2, 'northwest': 3, 'northeast': 4})

X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model_joblib_gb.pkl")
print("✅ Model trained and saved as model_joblib_gb.pkl")
