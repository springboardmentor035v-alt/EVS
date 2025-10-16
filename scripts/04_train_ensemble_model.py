import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import config # Uses the project's central configuration
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_ensemble_training():
    logging.info("🚀 Starting Module 4: Training ENSEMBLE Model")
    try:
        # Load the pre-processed and split data
        train_df = pd.read_csv(config.TRAIN_FILE)
        test_df = pd.read_csv(config.TEST_FILE)
    except FileNotFoundError as e:
        logging.error(f"❌ Error: {e}. Run previous pipeline scripts first.")
        return

    # Prepare the data for training
    available_features = [col for col in config.FEATURE_COLS if col in train_df.columns]
    X_train = train_df[available_features]
    y_train_raw = train_df[config.TARGET_COL]
    X_test = test_df[available_features]
    y_test_raw = test_df[config.TARGET_COL]
    
    # Encode the target labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    logging.info(f"✅ Target labels encoded. Classes: {le.classes_}")
    
    # --- YOUR ENSEMBLE MODEL LOGIC ---
    logging.info("🚀 Training Combined Voting Classifier (RandomForest + XGBoost)...")
    clf1 = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE)
    clf2 = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=config.RANDOM_STATE)

    ensemble_model = VotingClassifier(
        estimators=[('rf', clf1), ('xgb', clf2)],
        voting='hard' # Majority vote
    )
    
    ensemble_model.fit(X_train, y_train) # Note: uses non-scaled data as per friend's pipeline
    logging.info("✅ Ensemble model training complete.")

    # --- EVALUATION ---
    y_pred = ensemble_model.predict(X_test)
    report_str = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    
    # Save the evaluation report
    report_path = "outputs/ensemble_model_report.txt"
    with open(report_path, "w") as f:
        f.write("Ensemble Model (RF + XGBoost) Evaluation Report\n")
        f.write("="*50 + "\n")
        f.write(report_str)
    logging.info(f"✅ Classification report saved to '{report_path}'")
    print("\n--- Ensemble Model Performance ---")
    print(report_str)
    
    # --- SAVE THE MODEL AND ENCODER ---
    joblib.dump(ensemble_model, "outputs/ensemble_model.joblib")
    joblib.dump(le, config.ENCODER_FILE) # Overwrite with the same encoder for consistency
    logging.info("💾 Ensemble model and encoder saved to 'outputs/' directory.")
    logging.info("\n✅ Module 4 (Ensemble) completed successfully!")

if __name__ == "__main__":
    run_ensemble_training()