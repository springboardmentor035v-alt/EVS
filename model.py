import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')


class PollutionModelPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.encoder = None
        self.model = None

    def load_data(self):
        df = pd.read_csv(self.data_path)
        return df

    def clean_and_impute_data(self, df):
        df_clean = df.copy()
        pollution_cols = ['pm25', 'pm10', 'no2', 'co', 'so2', 'o3']
        value_limits = {
            'pm25': (0, 500),
            'pm10': (0, 1000),
            'no2': (0, 200),
            'co': (0, 5000),
            'so2': (0, 100),
            'o3': (0, 200)
        }
        for col, (min_val, max_val) in value_limits.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)
        for col in pollution_cols:
            if col in df_clean.columns:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
        return df_clean

    def categorize_pollution_source_simple(self, row):
        pm25 = row['pm25']
        pm10 = row['pm10']
        no2 = row['no2'] if not pd.isna(row['no2']) else 0
        so2 = row['so2'] if not pd.isna(row['so2']) else 0
        num_industrial = row.get('num_industrial', 0)
        num_farmland = row.get('num_farmland', 0)

        if so2 > 20 and num_industrial > 1:
            return 'industrial'
        elif no2 > 30:
            return 'vehicular'
        elif (pm25 > 60 or pm10 > 100) and num_farmland > 0:
            return 'agricultural'
        elif pm25 > 100 or pm10 > 200:
            return 'urban_high'
        elif pm25 > 40 or pm10 > 80:
            return 'urban_moderate'
        elif pm25 < 20 and pm10 < 40:
            return 'clean'
        else:
            return 'background'

    def create_final_dataset(self, df):
        final_features = [
            'pm25', 'pm10', 'no2', 'co', 'so2', 'o3',
            'latitude', 'longitude',
            'num_industrial', 'num_farmland', 'num_dumpsites', 'num_recycling',
            'weather',
            'pollution_source'
        ]
        final_df = df[final_features].copy()
        final_df['pm_ratio'] = final_df['pm25'] / (final_df['pm10'] + 1e-6)

        self.encoder = LabelEncoder()
        final_df['pollution_source_encoded'] = self.encoder.fit_transform(final_df['pollution_source'])

        weather_map = {'clear sky': 0, 'haze': 1, 'mist': 2, 'overcast clouds': 3,
                       'broken clouds': 4, 'scattered clouds': 5, 'light rain': 6,
                       'moderate rain': 7, 'thunderstorm': 8}
        final_df['weather_encoded'] = final_df['weather'].map(weather_map).fillna(9)
        return final_df

    def train_model(self, final_df):
        X = final_df.drop(['pollution_source', 'pollution_source_encoded', 'weather'], axis=1)
        y = final_df['pollution_source_encoded']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

        model = RandomForestClassifier(random_state=42, class_weight='balanced')

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )

        print("Starting hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("\nHyperparameter tuning complete.")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        class_names = self.encoder.classes_
        print(classification_report(y_test, y_pred, target_names=class_names))
        return self.model

    def save_pipeline(self):
        joblib.dump(self.model, './data/best_tuned_model.pkl')
        joblib.dump(self.encoder, './data/label_encoder.pkl')
        print("Model and encoder saved.")


if __name__ == "__main__":
    pipeline = PollutionModelPipeline('./data/module1_data_training.csv')
    df = pipeline.load_data()
    df_clean = pipeline.clean_and_impute_data(df)
    df_clean['pollution_source'] = df_clean.apply(pipeline.categorize_pollution_source_simple, axis=1)

    final_df = pipeline.create_final_dataset(df_clean)
    pipeline.train_model(final_df)
    pipeline.save_pipeline()
