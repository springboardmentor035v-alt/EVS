# scripts/04_train_model.py (FINAL VERSION 3.0)
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_model_training():
    logging.info("🚀 Starting Module 4: Model Training and Evaluation")
    try:
        train_df = pd.read_csv(config.TRAIN_FILE)
        test_df = pd.read_csv(config.TEST_FILE)
    except FileNotFoundError as e:
        logging.error(f"❌ Error: {e}. Run previous modules first.")
        return

    available_features = [col for col in config.FEATURE_COLS if col in train_df.columns]
    X_train = train_df[available_features]
    y_train_raw = train_df[config.TARGET_COL]
    X_test = test_df[available_features]
    y_test_raw = test_df[config.TARGET_COL]
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    logging.info(f"✅ Target labels encoded. Classes: {le.classes_}")
    
    num_classes = len(le.classes_)

    logging.info("🚀 Training XGBoost Classifier with GridSearchCV...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1]
    }

    # --- FINAL FIX: Change objective to 'multi:softmax' ---
    xgb = XGBClassifier(
        objective='multi:softmax',  # This was 'multi:softprob'
        num_class=num_classes, 
        eval_metric='mlogloss', 
        random_state=config.RANDOM_STATE
    )
    
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    logging.info(f"✅ GridSearchCV complete. Best parameters found: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    report_str = classification_report(
        y_test, y_pred, target_names=le.classes_, labels=range(len(le.classes_)), zero_division=0
    )
    
    with open(config.EVALUATION_FILE, "w") as f:
        f.write("XGBoost Model Evaluation Report\n")
        f.write("="*30 + "\n")
        f.write(f"Best Parameters: {grid_search.best_params_}\n\n")
        f.write(report_str)
    logging.info(f"✅ Classification report saved to '{config.EVALUATION_FILE}'")
    
    cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix for Best XGBoost Model')
    plt.savefig(config.CONFUSION_MATRIX_FILE)
    plt.close()

    feature_importances = pd.Series(best_model.feature_importances_, index=available_features).sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title('Feature Importance (XGBoost)')
    plt.savefig(config.FEATURE_IMPORTANCE_FILE)
    plt.close()
    
    joblib.dump(best_model, config.MODEL_FILE)
    joblib.dump(le, config.ENCODER_FILE)
    logging.info(f"💾 Best model and encoder saved.")
    logging.info("\n✅ Module 4 completed successfully!")

if __name__ == "__main__":
    run_model_training()