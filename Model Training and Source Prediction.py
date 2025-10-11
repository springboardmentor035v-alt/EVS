import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import joblib   # for saving the trained model

# --- Training & Prediction Function ---
def train_predict_models(df):
    # --- Features and Labels ---
    features = ['temperature', 'humidity', 'roads_count', 'factories_count',
                'infra_score', 'pollution_count']  # from cleaned + engineered dataset
    X = df[features]
    y = df['source']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=42)

    # --- Define Models ---
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Neural Network": MLPClassifier(max_iter=500, random_state=42)
    }

    # --- Hyperparameter grids ---
    param_grids = {
        "Logistic Regression": {"C": [0.1, 1, 10]},
        "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
        "Neural Network": {"hidden_layer_sizes": [(50,), (100,), (50,50)], "activation": ["relu", "tanh"]}
    }

    best_models = {}
    results = {}

    for name, model in models.items():
        print(f"\n🔹 Training {name}...")

        grid = GridSearchCV(model, param_grids[name], cv=3, scoring="f1_macro", n_jobs=-1)
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)

        # Save metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        best_models[name] = grid.best_estimator_

        print(f" {name} Results:")
        print(classification_report(y_test, y_pred))

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred, labels=y.unique())
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=y.unique(), yticklabels=y.unique())
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    # --- Choose best model by F1-score ---
    best_model_name = max(results, key=lambda x: results[x]["f1"])
    best_model = best_models[best_model_name]
    print(f"\n Best Model: {best_model_name} with F1-score {results[best_model_name]['f1']:.3f}")

    # Save model
    joblib.dump(best_model, "best_model.pkl")
    print("Model saved as best_model.pkl")

    return best_model, results


# --- Run training ---
df = pd.read_csv("train_balanced.csv")
best_model, metrics = train_predict_models(df)

print("\n Training Complete! Metrics Summary:")
print(pd.DataFrame(metrics))
