# scripts/model_training_optimized.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

# ----------------- LOAD DATA -----------------
INPUT_FILE = "data/labeled_data_new.csv"
df = pd.read_csv(INPUT_FILE)
print(f"🔹 Loaded labeled dataset: {df.shape}")

# ----------------- FEATURES & TARGET -----------------
# Select features
features = ["pm2_5","pm10","no2","so2","o3","co","temperature","humidity","wind_speed","latitude","longitude"]
X = df[features]

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(df["pollution_source"])  # Vehicular, Industrial, etc.
classes = le.classes_
print(f"🔹 Target classes: {list(classes)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------- TRAIN/TEST SPLIT -----------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"🔹 Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# ----------------- DECISION TREE -----------------
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("🔹 Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt, target_names=classes))

# ----------------- RANDOM FOREST WITH RANDOMIZEDSEARCHCV -----------------
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_params,
    n_iter=15,
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print("🔹 Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=classes))

# ----------------- XGBOOST WITH GRIDSEARCHCV -----------------
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    tree_method='hist',
    n_jobs=-1,
    eval_metric='mlogloss',
    random_state=42
)

xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

xgb_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_params,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1
)

xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
print("🔹 XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb, target_names=classes))

# ----------------- SAVE MODELS -----------------
joblib.dump(best_rf, "models/random_forest_model.joblib")
joblib.dump(best_xgb, "models/xgb_model.joblib")
joblib.dump(dt_model, "models/decision_tree_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(le, "models/label_encoder.joblib")

print("✅ Models, scaler, and label encoder saved to /models/")
