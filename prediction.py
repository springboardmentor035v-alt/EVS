# prediction.py
import joblib
import pandas as pd
import numpy as np

class PollutionPredictor:
    def __init__(self, model_path, encoder_path):
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model: {e}")

        try:
            self.encoder = joblib.load(encoder_path)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load encoder: {e}")

        self.features = ["pm25","pm10","no2","so2","co","o3",
                         "latitude","longitude","num_industrial",
                         "num_farmland","num_dumpsites","num_recycling","weather"]

    def predict(self, input_dict):
        try:
            df = pd.DataFrame([input_dict])

            # Encode categorical weather if encoder exists
            if "weather" in df.columns:
                try:
                    df["weather"] = self.encoder.transform(df["weather"])
                except Exception:
                    df["weather"] = 0  # fallback

            # Ensure all features exist
            for col in self.features:
                if col not in df.columns:
                    df[col] = 0

            X = df[self.features]

            # Get prediction & probability
            pred_class = self.model.predict(X)[0]
            if hasattr(self.model, "predict_proba"):
                conf = float(np.max(self.model.predict_proba(X)))
            else:
                conf = 1.0

            label = (self.encoder.inverse_transform([pred_class])[0]
                     if hasattr(self.encoder, "inverse_transform")
                     else str(pred_class))

            return {"pollution_source": label, "confidence": conf}

        except Exception as e:
            return {"pollution_source": "unknown", "confidence": 0.0, "error": str(e)}
