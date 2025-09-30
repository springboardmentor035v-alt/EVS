# scripts/04_train_model.py (FINAL CORRECTED VERSION)
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_model_training():
    logging.info("🚀 Starting Module 4: Model Training and Evaluation")

    try:
        train_df = pd.read_csv(config.TRAIN_FILE)
        test_df = pd.read_csv(config.TEST_FILE)
        logging.info("✅ Training and testing data loaded.")
    except FileNotFoundError as e:
        logging.error(f"❌ Error: {e}. Run previous modules first.")
        return

    available_features = [col for col in config.FEATURE_COLS if col in train_df.columns]
    logging.info(f"ℹ️ Using {len(available_features)} features for training: {available_features}")

    X_train = train_df[available_features]
    y_train_raw = train_df[config.TARGET_COL]
    X_test = test_df[available_features]
    y_test_raw = test_df[config.TARGET_COL]
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    logging.info(f"✅ Target labels encoded. Classes: {le.classes_}")
    
    logging.info("🌳 Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE, class_weight='balanced')
    model.fit(X_train, y_train)
    logging.info("✅ Model training complete.")

    y_pred = model.predict(X_test)
    
    logging.info("\n--- Model Performance Evaluation ---")
    
    # THIS IS THE FINAL FIX: Add 'labels' parameter to make the report robust
    # This prevents the ValueError if the small test set is missing a class by chance.
# THIS IS THE FINAL, CORRECT CODE
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=le.classes_, 
        labels=range(len(le.classes_)),  # <-- THIS IS THE MISSING LINE
        zero_division=0)
    logging.info(f"\nClassification Report:\n{report}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(config.CONFUSION_MATRIX_FILE)
    plt.close()
    logging.info(f"✅ Confusion matrix saved to '{config.CONFUSION_MATRIX_FILE}'")

    # Feature Importance
    feature_importances = pd.Series(model.feature_importances_, index=available_features).sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(config.FEATURE_IMPORTANCE_FILE)
    plt.close()
    logging.info(f"✅ Feature importance plot saved to '{config.FEATURE_IMPORTANCE_FILE}'")

    joblib.dump(model, config.MODEL_FILE)
    joblib.dump(le, config.ENCODER_FILE)
    logging.info(f"💾 Model and encoder saved.")
    logging.info("\n✅ Module 4 completed successfully!")

if __name__ == "__main__":
    run_model_training()