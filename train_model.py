import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load dataset
df = pd.read_csv("data/students.csv")

# Feature engineering
feature_cols = [
    "attendance_pct", "exam_score", "distance_km", "midday_meal",
    "sibling_dropout", "prev_year_score", "teacher_engagement", "extracurricular"
]
le_income = LabelEncoder()
le_residence = LabelEncoder()
df["family_income_enc"] = le_income.fit_transform(df["family_income"])
df["residence_enc"] = le_residence.fit_transform(df["residence"])
feature_cols += ["family_income_enc", "residence_enc"]

X = df[feature_cols]
y_score = df["risk_score"]
y_label = df["risk_label"].map({"Low": 0, "Medium": 1, "High": 2})

# Train Regressor for live risk score
X_train, X_test, y_train, y_test = train_test_split(X, y_score, test_size=0.2, random_state=42)
regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Train Classifier
clf_X_train, clf_X_test, clf_y_train, clf_y_test = train_test_split(X, y_label, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(clf_X_train, clf_y_train)

# Save models
os.makedirs("models", exist_ok=True)
with open("models/regressor.pkl", "wb") as f:
    pickle.dump(regressor, f)
with open("models/classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)
with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump({"income": le_income, "residence": le_residence}, f)
with open("models/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print("Models trained and saved.")
print(f"Regressor R2 score: {regressor.score(X_test, y_test):.4f}")
print(f"Classifier Accuracy: {classifier.score(clf_X_test, clf_y_test):.4f}")
print(f"Feature importances: {dict(zip(feature_cols, regressor.feature_importances_.round(3)))}")
