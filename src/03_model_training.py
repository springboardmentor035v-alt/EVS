import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --- CONFIGURATION ---
DATA_PATH = 'data/labeled_predictions.csv' 
MODEL_SAVE_PATH = 'model/random_forest_model.pkl'
TARGET_COLUMN = 'Pollution_Source' 
FEATURE_COLUMNS = ['PM25', 'NO2', 'Wind_Speed', 'Temp'] # IMPORTANT: Match these to your CSV header!

def train_model():
    # 1. Load Data
    try:
        # We assume your mock data has the required columns
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}. Check file path and column names.")
        return

    # 2. Prepare Data (Only use the columns defined in FEATURE_COLUMNS)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Model requires all features to be numeric, and your mock data is already set up this way.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Model Training
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Training complete.")

    # 4. Evaluation (CRITICAL requirement for the report)
    y_pred = model.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 5. Save Model and Generate Predictions for Dashboard
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    # Generate predictions on the full dataset
    full_predictions = model.predict(X)
    df['Predicted_Source'] = full_predictions
    # Get confidence score (max probability of the predicted class)
    df['Confidence_Score'] = model.predict_proba(X).max(axis=1) 
    df['Pollution_Level'] = df['Pollution_Level'].round(0) # Ensure a clean numeric column

    # Save the final dataset the dashboard will load (CRITICAL)
    # This overwrites the original CSV, adding the two prediction columns.
    df.to_csv(DATA_PATH, index=False)
    print(f"Predictions and Confidence Scores added to {DATA_PATH}")

if __name__ == '__main__':
    train_model()