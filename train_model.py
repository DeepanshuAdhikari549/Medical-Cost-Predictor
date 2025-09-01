# train_model.py  (consistent with your Streamlit features)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn import __version__ as skver
import joblib

print("Using scikit-learn:", skver)

# 1) Load
df = pd.read_csv("insurance.csv")

# 2) Encode EXACTLY like app uses
df["sex"] = df["sex"].map({"male": 1, "female": 0})
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
df["region"] = df["region"].map({"southwest": 1, "southeast": 2, "northwest": 3, "northeast": 4})

# 3) Features in fixed order
X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["charges"]

# 4) Split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 5) Train
gb = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
)
gb.fit(X_tr, y_tr)

# 6) Evaluate
pred = gb.predict(X_te)
print("MAE:", mean_absolute_error(y_te, pred))
print("R2 :", r2_score(y_te, pred))

# 7) Save (joblib)
joblib.dump(gb, "model_joblib_gb.pkl")
print("✅ Saved model_joblib_gb.pkl with sklearn", skver)
