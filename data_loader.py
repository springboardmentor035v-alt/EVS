import pandas as pd
import numpy as np
import requests
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib

import folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# For API calls and location features
try:
    import osmnx as ox
    from geopy.distance import geodesic
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    print("Warning: osmnx or geopy not available. Some geospatial features may be limited.")

def fetch_openaq_data(city, params=None):
    """
    Fetch pollution data from OpenAQ API
    
    Args:
        city (str): City name
        params (dict): Additional API parameters
    
    Returns:
        pd.DataFrame: Pollution measurements data
    """
    try:
        url = f"https://api.openaq.org/v2/measurements"
        default_params = {
            'city': city,
            'limit': 1000,
            'has_geo': 'true'
        }
        if params:
            default_params.update(params)
            
        response = requests.get(url, params=default_params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'results' in data and data['results']:
            return pd.DataFrame(data['results'])
        else:
            print(f"No data found for city: {city}")
            return create_sample_data()
    except requests.RequestException as e:
        print(f"Error fetching data from OpenAQ API: {e}")
        print("Using sample data instead...")
        return create_sample_data()

def create_sample_data():
    """Create sample pollution data for testing purposes"""
    np.random.seed(42)
    n_samples = 500
    
    # Generate sample coordinates around a city center (e.g., Delhi)
    center_lat, center_lon = 28.6139, 77.2090
    
    data = {
        'parameter': np.random.choice(['pm25', 'no2', 'so2', 'co', 'o3'], n_samples),
        'value': np.random.exponential(scale=50, size=n_samples),
        'unit': ['µg/m³'] * n_samples,
        'date': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'coordinates': {
            'latitude': center_lat + np.random.normal(0, 0.1, n_samples),
            'longitude': center_lon + np.random.normal(0, 0.1, n_samples)
        },
        'city': ['Delhi'] * n_samples,
        'location': [f'Station_{i%50}' for i in range(n_samples)]
    }
    
    return pd.DataFrame(data)

def fetch_weather_data(lat, lon, api_key=None):
    """
    Fetch weather data from OpenWeatherMap API
    
    Args:
        lat (float): Latitude
        lon (float): Longitude  
        api_key (str): OpenWeatherMap API key
    
    Returns:
        dict: Weather data
    """
    if not api_key:
        # Return sample weather data if no API key
        return {
            'main': {
                'temp': 25.0 + np.random.normal(0, 5),
                'humidity': 60 + np.random.normal(0, 15),
                'pressure': 1013 + np.random.normal(0, 20)
            },
            'wind': {
                'speed': 3.0 + np.random.normal(0, 1),
                'deg': np.random.randint(0, 360)
            }
        }
    
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat, 
            'lon': lon, 
            'appid': api_key,
            'units': 'metric'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return fetch_weather_data(lat, lon)  # Return sample data

def get_location_features(lat, lon, dist=1000):
    """
    Calculate proximity features using OpenStreetMap data
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        dist (int): Distance radius in meters
    
    Returns:
        dict: Location features including road and factory proximity
    """
    if not GEOSPATIAL_AVAILABLE:
        # Return sample proximity data
        return {
            'roads_proximity': np.random.exponential(100),
            'factories_proximity': np.random.exponential(500),
            'near_main_road': np.random.choice([0, 1]),
            'near_factory': np.random.choice([0, 1]),
            'near_farmland': np.random.choice([0, 1])
        }
    
    try:
        # Get road network
        G = ox.graph_from_point((lat, lon), dist=dist, network_type='drive')
        roads = ox.geometries.geometries_from_point((lat, lon), tags={'highway': True}, dist=dist)
        
        # Get industrial areas
        factories = ox.geometries.geometries_from_point(
            (lat, lon), 
            tags={'landuse': 'industrial'}, 
            dist=dist
        )
        
        # Calculate proximities (simplified)
        roads_proximity = len(roads) * 10 if not roads.empty else 1000
        factories_proximity = len(factories) * 50 if not factories.empty else 1000
        
        return {
            'roads_proximity': min(roads_proximity, 1000),
            'factories_proximity': min(factories_proximity, 2000),
            'near_main_road': 1 if roads_proximity < 200 else 0,
            'near_factory': 1 if factories_proximity < 500 else 0,
            'near_farmland': np.random.choice([0, 1])  # Simplified
        }
    except Exception as e:
        print(f"Error calculating location features: {e}")
        return get_location_features(lat, lon, dist)  # Return sample data

"""Data Cleaning and Feature Engineering"""

def clean_pollution_data(df):
    """
    Comprehensive data cleaning for pollution dataset
    
    Args:
        df (pd.DataFrame): Raw pollution data
    
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("Starting data cleaning...")
    original_shape = df.shape
    
    # Handle coordinates structure
    if 'coordinates' in df.columns and isinstance(df['coordinates'].iloc[0], dict):
        df['latitude'] = df['coordinates'].apply(lambda x: x.get('latitude') if isinstance(x, dict) else np.nan)
        df['longitude'] = df['coordinates'].apply(lambda x: x.get('longitude') if isinstance(x, dict) else np.nan)
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    # Handle missing values in critical columns
    critical_cols = ['value', 'latitude', 'longitude']
    for col in critical_cols:
        if col in df.columns:
            df = df.dropna(subset=[col])
    
    # Convert data types
    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['value'])
    
    # Handle timestamp
    if 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['timestamp'] = pd.date_range('2024-01-01', periods=len(df), freq='H')
    
    # Remove outliers (values beyond 3 standard deviations)
    if 'value' in df.columns:
        Q1 = df['value'].quantile(0.25)
        Q3 = df['value'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_before = len(df)
        df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]
        print(f"Removed {outliers_before - len(df)} outliers")
    
    # Impute remaining missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    print(f"Data cleaning completed. Final shape: {df.shape}")
    return df

def feature_engineering(df):
    """
    Comprehensive feature engineering for pollution data
    
    Args:
        df (pd.DataFrame): Cleaned pollution data
        
    Returns:
        pd.DataFrame: Feature-engineered dataset
    """
    print("Starting feature engineering...")
    
    # Create pollutant-specific columns
    if 'parameter' in df.columns:
        pollutant_df = df.pivot_table(
            index=['latitude', 'longitude', 'timestamp'], 
            columns='parameter', 
            values='value', 
            aggfunc='mean'
        ).reset_index()
        
        # Fill missing pollutant values with median
        for col in pollutant_df.columns:
            if col not in ['latitude', 'longitude', 'timestamp']:
                pollutant_df[col] = pollutant_df[col].fillna(pollutant_df[col].median())
        
        df = pollutant_df
    
    # Ensure required pollutant columns exist
    required_pollutants = ['pm25', 'no2', 'so2', 'co']
    for pollutant in required_pollutants:
        if pollutant not in df.columns:
            # Create synthetic data based on correlations
            if 'pm25' in df.columns and pollutant != 'pm25':
                correlation_factor = {'no2': 0.7, 'so2': 0.5, 'co': 0.6}.get(pollutant, 0.5)
                df[pollutant] = df['pm25'] * correlation_factor + np.random.normal(0, 5, len(df))
            else:
                df[pollutant] = np.random.exponential(scale=30, size=len(df))
    
    # Rename columns for consistency
    column_mapping = {
        'pm25': 'PM2.5',
        'no2': 'NO2', 
        'so2': 'SO2',
        'co': 'CO'
    }
    df = df.rename(columns=column_mapping)
    
    # Add weather features
    weather_features = []
    for idx, row in df.iterrows():
        weather = fetch_weather_data(row['latitude'], row['longitude'])
        weather_features.append({
            'temperature': weather['main']['temp'],
            'humidity': weather['main']['humidity'],
            'pressure': weather['main']['pressure'],
            'wind_speed': weather['wind']['speed']
        })
    
    weather_df = pd.DataFrame(weather_features)
    df = pd.concat([df, weather_df], axis=1)
    
    # Add location proximity features
    location_features = []
    for idx, row in df.iterrows():
        loc_features = get_location_features(row['latitude'], row['longitude'])
        location_features.append(loc_features)
    
    location_df = pd.DataFrame(location_features)
    df = pd.concat([df, location_df], axis=1)
    
    # Temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['season'] = df['month'].apply(lambda x: 'Winter' if x in [12, 1, 2] 
                                    else 'Spring' if x in [3, 4, 5]
                                    else 'Summer' if x in [6, 7, 8] 
                                    else 'Fall')
    
    # Normalize numerical features
    scaler = StandardScaler()
    numeric_columns = ['PM2.5', 'NO2', 'SO2', 'CO', 'temperature', 'humidity', 'pressure', 'wind_speed']
    
    for col in numeric_columns:
        if col in df.columns:
            df[f'{col}_normalized'] = scaler.fit_transform(df[[col]])
    
    # Create composite pollution index
    if all(col in df.columns for col in ['PM2.5', 'NO2', 'SO2', 'CO']):
        df['pollution_index'] = (df['PM2.5'] * 0.4 + df['NO2'] * 0.3 + 
                                df['SO2'] * 0.2 + df['CO'] * 0.1)
    
    print(f"Feature engineering completed. Final shape: {df.shape}")
    return df

"""Source Labelling and Simulation"""

def label_sources(df):
    """
    Comprehensive source labeling based on pollution patterns and location features
    
    Args:
        df (pd.DataFrame): Feature-engineered dataset
        
    Returns:
        pd.DataFrame: Dataset with source labels
    """
    print("Starting source labeling...")
    
    # Initialize source column
    df['source'] = 'Unknown'
    
    # Rule-based labeling with multiple criteria
    
    # Vehicular sources - high NO2, near roads, peak traffic hours
    vehicular_mask = (
        (df['NO2'] > df['NO2'].quantile(0.7)) & 
        (df['near_main_road'] == 1) &
        (df['hour'].isin([7, 8, 9, 17, 18, 19, 20]))  # Rush hours
    )
    df.loc[vehicular_mask, 'source'] = 'Vehicular'
    
    # Industrial sources - high SO2, near factories, weekdays
    industrial_mask = (
        (df['SO2'] > df['SO2'].quantile(0.6)) &
        (df['near_factory'] == 1) &
        (df['is_weekend'] == 0) &
        (df['hour'].between(8, 18))  # Working hours
    )
    df.loc[industrial_mask, 'source'] = 'Industrial'
    
    # Agricultural sources - high PM2.5, near farmland, dry season
    agricultural_mask = (
        (df['PM2.5'] > df['PM2.5'].quantile(0.8)) &
        (df['near_farmland'] == 1) &
        (df['season'].isin(['Summer', 'Fall'])) &
        (df['humidity'] < df['humidity'].quantile(0.3))
    )
    df.loc[agricultural_mask, 'source'] = 'Agricultural'
    
    # Residential sources - moderate pollution, not near major sources
    residential_mask = (
        (df['source'] == 'Unknown') &
        (df['near_main_road'] == 0) &
        (df['near_factory'] == 0) &
        (df['pollution_index'] > df['pollution_index'].quantile(0.3)) &
        (df['pollution_index'] < df['pollution_index'].quantile(0.7))
    )
    df.loc[residential_mask, 'source'] = 'Residential'
    
    # Natural sources - low pollution in remote areas
    natural_mask = (
        (df['source'] == 'Unknown') &
        (df['pollution_index'] < df['pollution_index'].quantile(0.3)) &
        (df['roads_proximity'] > 500) &
        (df['factories_proximity'] > 1000)
    )
    df.loc[natural_mask, 'source'] = 'Natural'
    
    # Mixed sources for remaining unknown
    df.loc[df['source'] == 'Unknown', 'source'] = 'Mixed'
    
    # Balance the dataset if needed
    source_counts = df['source'].value_counts()
    print("Source distribution before balancing:")
    print(source_counts)
    
    # Simulate additional labels for better balance
    min_samples = max(50, source_counts.min())  # Ensure minimum samples
    for source in source_counts.index:
        current_count = source_counts[source]
        if current_count < min_samples:
            # Generate synthetic samples
            source_data = df[df['source'] == source].copy()
            needed_samples = min_samples - current_count
            
            # Create synthetic samples with slight variations
            synthetic_samples = []
            for _ in range(needed_samples):
                base_sample = source_data.sample(1).copy()
                # Add small random variations
                for col in ['PM2.5', 'NO2', 'SO2', 'CO']:
                    if col in base_sample.columns:
                        base_sample[col] *= (1 + np.random.normal(0, 0.1))
                synthetic_samples.append(base_sample)
            
            if synthetic_samples:
                synthetic_df = pd.concat(synthetic_samples, ignore_index=True)
                df = pd.concat([df, synthetic_df], ignore_index=True)
    
    print("Source distribution after balancing:")
    print(df['source'].value_counts())
    
    return df

def split_dataset(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split dataset into training, validation, and testing sets with stratification
    
    Args:
        df (pd.DataFrame): Dataset with source labels
        test_size (float): Proportion for test set
        val_size (float): Proportion for validation set  
        random_state (int): Random seed
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print("Splitting dataset...")
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['source'],
        random_state=random_state
    )
    
    # Second split: train vs validation
    val_proportion = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_proportion,
        stratify=train_val_df['source'], 
        random_state=random_state
    )
    
    print(f"Dataset split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def apply_data_balancing(X_train, y_train):
    """
    Apply SMOTE for handling class imbalance
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        tuple: Balanced (X_train, y_train)
    """
    print("Applying SMOTE for data balancing...")
    
    try:
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"Original distribution: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"Balanced distribution: {pd.Series(y_balanced).value_counts().to_dict()}")
        
        return X_balanced, y_balanced
    except Exception as e:
        print(f"SMOTE failed: {e}. Using original data.")
        return X_train, y_train

"""Model Training and Source Prediction"""

def prepare_features(df):
    """
    Prepare feature matrix for model training
    
    Args:
        df (pd.DataFrame): Dataset with all features
        
    Returns:
        tuple: (X, feature_names)
    """
    feature_columns = [
        'PM2.5', 'NO2', 'SO2', 'CO',
        'temperature', 'humidity', 'pressure', 'wind_speed',
        'roads_proximity', 'factories_proximity',
        'near_main_road', 'near_factory', 'near_farmland',
        'hour', 'dayofweek', 'month', 'is_weekend',
        'pollution_index'
    ]
    
    # Select only available columns
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features]
    
    # Handle any remaining missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=available_features)
    
    return X, available_features

def train_multiple_models(X_train, y_train, X_val, y_val):
    """
    Train multiple ML models and compare performance
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        dict: Trained models with performance metrics
    """
    print("Training multiple models...")
    
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
    }
    
    # Parameter grids for hyperparameter tuning
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        },
        'Logistic Regression': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'adaptive']
        }
    }
    
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Hyperparameter tuning with GridSearchCV
        param_grid = param_grids[model_name]
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=3, 
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Validate on validation set
        y_val_pred = best_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1_weighted')
        
        trained_models[model_name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'val_accuracy': val_accuracy,
            'val_f1_score': val_f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_val_pred
        }
        
        print(f"{model_name} - Val Accuracy: {val_accuracy:.3f}, Val F1: {val_f1:.3f}")
        print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return trained_models

def evaluate_model_performance(models, X_test, y_test):
    """
    Comprehensive model evaluation on test set
    
    Args:
        models (dict): Trained models
        X_test, y_test: Test data
        
    Returns:
        dict: Performance metrics and visualizations
    """
    print("\nEvaluating models on test set...")
    
    performance_results = {}
    
    for model_name, model_info in models.items():
        model = model_info['model']
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        performance_results[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        print(f"\n{model_name} Test Results:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    return performance_results

def create_performance_visualizations(performance_results, y_test):
    """
    Create performance visualization plots
    
    Args:
        performance_results (dict): Model performance metrics
        y_test: True test labels
    """
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    # 1. Model comparison bar plot
    plt.subplot(2, 3, 1)
    models = list(performance_results.keys())
    accuracies = [performance_results[m]['accuracy'] for m in models]
    f1_scores = [performance_results[m]['f1_score'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    plt.bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2-4. Confusion matrices for each model
    for i, (model_name, results) in enumerate(performance_results.items()):
        plt.subplot(2, 3, i + 2)
        cm = results['confusion_matrix']
        labels = sorted(set(y_test))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'{model_name}\nConfusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('/tmp/model_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_best_model(trained_models, performance_results, feature_names):
    """
    Save the best performing model
    
    Args:
        trained_models (dict): All trained models
        performance_results (dict): Performance metrics
        feature_names (list): Feature column names
        
    Returns:
        str: Best model name
    """
    # Find best model based on test F1 score
    best_model_name = max(performance_results.keys(), 
                         key=lambda x: performance_results[x]['f1_score'])
    
    best_model = trained_models[best_model_name]['model']
    
    # Save model and metadata
    model_data = {
        'model': best_model,
        'model_name': best_model_name,
        'feature_names': feature_names,
        'performance': performance_results[best_model_name],
        'hyperparameters': trained_models[best_model_name]['best_params']
    }
    
    joblib.dump(model_data, '/tmp/best_pollution_source_model.pkl')
    
    print(f"\nBest model ({best_model_name}) saved with F1 score: {performance_results[best_model_name]['f1_score']:.3f}")
    
    return best_model_name

def predict_pollution_sources(df, model_path='/tmp/best_pollution_source_model.pkl'):
    """
    Make predictions using the trained model
    
    Args:
        df (pd.DataFrame): Data for prediction
        model_path (str): Path to saved model
        
    Returns:
        pd.DataFrame: Data with predictions and confidence scores
    """
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        # Prepare features
        X = df[feature_names]
        
        # Make predictions
        predictions = model.predict(X)
        prediction_probabilities = model.predict_proba(X)
        confidence_scores = np.max(prediction_probabilities, axis=1)
        
        # Add predictions to dataframe
        df_pred = df.copy()
        df_pred['predicted_source'] = predictions
        df_pred['confidence_score'] = confidence_scores
        
        return df_pred
        
    except Exception as e:
        print(f"Error making predictions: {e}")
        return df

"""Geospatial Mapping and Heatmap Visualization"""

def create_pollution_heatmap(df, pollutant='PM2.5', save_path='/tmp/pollution_heatmap.html'):
    """
    Create interactive heatmap visualization for pollution data
    
    Args:
        df (pd.DataFrame): Data with pollution measurements and coordinates
        pollutant (str): Pollutant to visualize
        save_path (str): Path to save the HTML file
        
    Returns:
        folium.Map: Interactive map object
    """
    print(f"Creating pollution heatmap for {pollutant}...")
    
    # Calculate center point
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Prepare heatmap data
    if pollutant in df.columns:
        heat_data = [[row['latitude'], row['longitude'], row[pollutant]] 
                    for idx, row in df.iterrows() 
                    if pd.notna(row['latitude']) and pd.notna(row['longitude']) and pd.notna(row[pollutant])]
        
        # Add heatmap layer
        HeatMap(heat_data, radius=15, blur=10, gradient={
            0.0: 'blue',
            0.3: 'cyan', 
            0.5: 'lime',
            0.7: 'yellow',
            1.0: 'red'
        }).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    m.save(save_path)
    print(f"Heatmap saved to {save_path}")
    
    return m

def create_source_map(df, save_path='/tmp/source_map.html'):
    """
    Create map with source-specific markers
    
    Args:
        df (pd.DataFrame): Data with source predictions and coordinates
        save_path (str): Path to save the HTML file
        
    Returns:
        folium.Map: Interactive map with source markers
    """
    print("Creating source-specific map...")
    
    # Calculate center point
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Source color mapping
    source_colors = {
        'Vehicular': 'red',
        'Industrial': 'darkred', 
        'Agricultural': 'green',
        'Residential': 'blue',
        'Natural': 'lightgreen',
        'Mixed': 'purple',
        'Unknown': 'gray'
    }
    
    # Source icon mapping  
    source_icons = {
        'Vehicular': 'car',
        'Industrial': 'industry',
        'Agricultural': 'leaf',
        'Residential': 'home',
        'Natural': 'tree',
        'Mixed': 'question',
        'Unknown': 'question'
    }
    
    # Add markers for each source type
    source_col = 'predicted_source' if 'predicted_source' in df.columns else 'source'
    
    for source in df[source_col].unique():
        source_data = df[df[source_col] == source]
        
        # Create feature group for this source
        feature_group = folium.FeatureGroup(name=f'{source} Sources')
        
        for idx, row in source_data.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                # Create popup text
                popup_text = f"""
                <b>Source:</b> {row[source_col]}<br>
                <b>Location:</b> ({row['latitude']:.4f}, {row['longitude']:.4f})<br>
                """
                
                if 'PM2.5' in row:
                    popup_text += f"<b>PM2.5:</b> {row['PM2.5']:.2f} µg/m³<br>"
                if 'NO2' in row:
                    popup_text += f"<b>NO2:</b> {row['NO2']:.2f} µg/m³<br>"
                if 'confidence_score' in row:
                    popup_text += f"<b>Confidence:</b> {row['confidence_score']:.2f}<br>"
                
                # Add marker
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"{source}: {row.get('pollution_index', 'N/A'):.1f}",
                    icon=folium.Icon(
                        color=source_colors.get(source, 'gray'),
                        icon=source_icons.get(source, 'question'),
                        prefix='fa'
                    )
                ).add_to(feature_group)
        
        feature_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Pollution Sources</b></p>
    '''
    
    for source, color in source_colors.items():
        legend_html += f'<p><i class="fa fa-circle" style="color:{color}"></i> {source}</p>'
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(save_path)
    print(f"Source map saved to {save_path}")
    
    return m

def create_multi_layer_map(df, save_path='/tmp/multi_layer_map.html'):
    """
    Create comprehensive map with multiple data layers
    
    Args:
        df (pd.DataFrame): Complete dataset
        save_path (str): Path to save the HTML file
        
    Returns:
        folium.Map: Multi-layer interactive map
    """
    print("Creating multi-layer comprehensive map...")
    
    # Calculate center point
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    # Create base map with multiple tile options
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # Add alternative tile layers
    folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
    
    # Layer 1: PM2.5 Heatmap
    if 'PM2.5' in df.columns:
        pm25_data = [[row['latitude'], row['longitude'], row['PM2.5']] 
                    for idx, row in df.iterrows() 
                    if pd.notna(row['latitude']) and pd.notna(row['longitude']) and pd.notna(row['PM2.5'])]
        
        pm25_heatmap = folium.FeatureGroup(name='PM2.5 Heatmap')
        HeatMap(pm25_data, radius=12, blur=8, gradient={
            0.0: 'blue', 0.3: 'cyan', 0.5: 'lime', 0.7: 'yellow', 1.0: 'red'
        }).add_to(pm25_heatmap)
        pm25_heatmap.add_to(m)
    
    # Layer 2: High pollution zones (circles)
    if 'pollution_index' in df.columns:
        high_pollution = df[df['pollution_index'] > df['pollution_index'].quantile(0.8)]
        
        pollution_zones = folium.FeatureGroup(name='High Pollution Zones')
        for idx, row in high_pollution.iterrows():
            folium.Circle(
                location=[row['latitude'], row['longitude']],
                radius=200,
                popup=f"High Pollution: {row['pollution_index']:.2f}",
                color='red',
                fill=True,
                fillOpacity=0.3
            ).add_to(pollution_zones)
        pollution_zones.add_to(m)
    
    # Layer 3: Source markers (from previous function)
    source_col = 'predicted_source' if 'predicted_source' in df.columns else 'source'
    source_colors = {
        'Vehicular': 'red', 'Industrial': 'darkred', 'Agricultural': 'green',
        'Residential': 'blue', 'Natural': 'lightgreen', 'Mixed': 'purple', 'Unknown': 'gray'
    }
    
    sources_layer = folium.FeatureGroup(name='Pollution Sources')
    for idx, row in df.sample(min(100, len(df))).iterrows():  # Sample for performance
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=f"Source: {row[source_col]}",
                color=source_colors.get(row[source_col], 'gray'),
                fill=True,
                fillOpacity=0.7
            ).add_to(sources_layer)
    sources_layer.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    m.save(save_path)
    print(f"Multi-layer map saved to {save_path}")
    
    return m

def generate_pollution_insights(df):
    """
    Generate insights and statistics from pollution data
    
    Args:
        df (pd.DataFrame): Complete dataset with predictions
        
    Returns:
        dict: Insights and statistics
    """
    insights = {}
    
    # Basic statistics
    if 'pollution_index' in df.columns:
        insights['avg_pollution'] = df['pollution_index'].mean()
        insights['max_pollution'] = df['pollution_index'].max()
        insights['pollution_std'] = df['pollution_index'].std()
    
    # Source distribution
    source_col = 'predicted_source' if 'predicted_source' in df.columns else 'source'
    insights['source_distribution'] = df[source_col].value_counts().to_dict()
    
    # Temporal patterns
    if 'hour' in df.columns:
        insights['peak_hours'] = df.groupby('hour')['pollution_index'].mean().idxmax()
    
    # Spatial patterns
    insights['geographic_bounds'] = {
        'north': df['latitude'].max(),
        'south': df['latitude'].min(), 
        'east': df['longitude'].max(),
        'west': df['longitude'].min()
    }
    
    # High-risk areas
    if 'pollution_index' in df.columns:
        high_risk_threshold = df['pollution_index'].quantile(0.9)
        high_risk_areas = df[df['pollution_index'] > high_risk_threshold]
        insights['high_risk_locations'] = len(high_risk_areas)
        insights['high_risk_percentage'] = (len(high_risk_areas) / len(df)) * 100
    
    return insights


"""Real-Time Dashboard and Alerts (Streamlit)"""

def create_streamlit_app():
    """
    Create comprehensive Streamlit dashboard code
    
    Returns:
        str: Complete Streamlit application code
    """
    
    app_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="EnviroScan - Pollution Source Identifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🌍 EnviroScan Pollution Source Identifier</h1>', 
           unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("🔧 Configuration")

# City input
city = st.sidebar.text_input("📍 Enter City Name", value="Delhi", key="city_input")

# Coordinate inputs
st.sidebar.subheader("📍 Custom Coordinates (Optional)")
use_coordinates = st.sidebar.checkbox("Use custom coordinates")

if use_coordinates:
    latitude = st.sidebar.number_input("Latitude", value=28.6139, format="%.4f")
    longitude = st.sidebar.number_input("Longitude", value=77.2090, format="%.4f") 
else:
    latitude, longitude = 28.6139, 77.2090  # Default to Delhi

# Time range selection
st.sidebar.subheader("📅 Time Range")
time_range = st.sidebar.selectbox(
    "Select time period",
    ["Last 24 hours", "Last 7 days", "Last 30 days", "Custom range"]
)

if time_range == "Custom range":
    start_date = st.sidebar.date_input("Start date", datetime.now() - timedelta(days=7))
    end_date = st.sidebar.date_input("End date", datetime.now())

# Pollution source filter
st.sidebar.subheader("🏭 Source Filter")
source_filter = st.sidebar.multiselect(
    "Select pollution sources",
    ["All", "Vehicular", "Industrial", "Agricultural", "Residential", "Natural"],
    default=["All"]
)

# Analysis button
analyze_button = st.sidebar.button("🔍 Analyze Pollution", type="primary")

@st.cache_data
def generate_sample_data(city_name, lat, lon, n_samples=200):
    """Generate sample pollution data for demonstration"""
    np.random.seed(42)
    
    # Generate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    timestamps = pd.date_range(start_time, end_time, periods=n_samples)
    
    # Generate coordinates around the specified location
    lats = lat + np.random.normal(0, 0.05, n_samples)
    lons = lon + np.random.normal(0, 0.05, n_samples)
    
    # Generate pollution data with realistic patterns
    hours = [t.hour for t in timestamps]
    base_pm25 = 30 + 20 * np.sin(np.array(hours) * np.pi / 12)  # Daily pattern
    base_no2 = 25 + 15 * np.sin(np.array(hours) * np.pi / 12 + np.pi/4)
    
    data = {
        'timestamp': timestamps,
        'latitude': lats,
        'longitude': lons,
        'PM2.5': np.maximum(base_pm25 + np.random.normal(0, 10, n_samples), 0),
        'NO2': np.maximum(base_no2 + np.random.normal(0, 8, n_samples), 0),
        'SO2': np.maximum(np.random.exponential(15, n_samples), 0),
        'CO': np.maximum(np.random.exponential(1.2, n_samples), 0),
        'temperature': 25 + 10 * np.sin(np.array(hours) * np.pi / 12) + np.random.normal(0, 3, n_samples),
        'humidity': 60 + 20 * np.random.random(n_samples),
        'source': np.random.choice(['Vehicular', 'Industrial', 'Agricultural', 'Residential'], n_samples),
        'confidence': 0.7 + 0.3 * np.random.random(n_samples)
    }
    
    df = pd.DataFrame(data)
    df['pollution_index'] = (df['PM2.5'] * 0.4 + df['NO2'] * 0.3 + df['SO2'] * 0.2 + df['CO'] * 0.1)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    
    return df

def get_aqi_level(pollution_index):
    """Determine AQI level and color"""
    if pollution_index <= 50:
        return "Good", "#4CAF50", "😊"
    elif pollution_index <= 100:
        return "Moderate", "#FFC107", "😐"  
    elif pollution_index <= 150:
        return "Unhealthy for Sensitive Groups", "#FF9800", "😷"
    elif pollution_index <= 200:
        return "Unhealthy", "#F44336", "😨"
    else:
        return "Very Unhealthy", "#9C27B0", "💀"

def create_alert(pollution_index, source):
    """Create pollution alert based on levels"""
    level, color, emoji = get_aqi_level(pollution_index)
    
    if pollution_index > 150:
        alert_class = "alert-high"
        urgency = "🚨 CRITICAL ALERT"
    elif pollution_index > 100:
        alert_class = "alert-medium"  
        urgency = "⚠️ WARNING"
    else:
        alert_class = "alert-low"
        urgency = "✅ NORMAL"
    
    recommendation = ("Avoid outdoor activities. Use air purifiers indoors." if pollution_index > 150 
                     else "Limit prolonged outdoor exposure." if pollution_index > 100
                     else "Air quality is acceptable for most people.")
    
    st.markdown(f"""
    <div class="{alert_class}">
        <h4>{urgency}</h4>
        <p><strong>Current AQI Level:</strong> {level} ({pollution_index:.1f}) {emoji}</p>
        <p><strong>Primary Source:</strong> {source}</p>
        <p><strong>Recommendation:</strong> {recommendation}</p>
    </div>
    """, unsafe_allow_html=True)

# Main analysis section
if analyze_button or city:
    with st.spinner(f"🔄 Analyzing pollution data for {city}..."):
        # Generate or load data
        df = generate_sample_data(city, latitude, longitude)
        
        # Apply source filter
        if "All" not in source_filter:
            df = df[df['source'].isin(source_filter)]
        
        # Current pollution status
        st.header("📊 Current Pollution Status")
        
        latest_data = df.iloc[-1]
        current_pollution = latest_data['pollution_index']
        primary_source = latest_data['source']
        
        # Create alert
        create_alert(current_pollution, primary_source)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🌫️ PM2.5",
                value=f"{latest_data['PM2.5']:.1f} µg/m³",
                delta=f"{np.random.uniform(-5, 5):.1f}"
            )
        
        with col2:
            st.metric(
                label="🚗 NO2", 
                value=f"{latest_data['NO2']:.1f} µg/m³",
                delta=f"{np.random.uniform(-3, 3):.1f}"
            )
        
        with col3:
            st.metric(
                label="🏭 SO2",
                value=f"{latest_data['SO2']:.1f} µg/m³", 
                delta=f"{np.random.uniform(-2, 2):.1f}"
            )
        
        with col4:
            st.metric(
                label="🔥 CO",
                value=f"{latest_data['CO']:.2f} mg/m³",
                delta=f"{np.random.uniform(-0.5, 0.5):.2f}"
            )
        
        # Charts section
        st.header("📈 Pollution Trends")
        
        # Time series chart
        fig_timeline = px.line(
            df, 
            x='timestamp', 
            y=['PM2.5', 'NO2', 'SO2'], 
            title="Pollutant Levels Over Time",
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        fig_timeline.update_layout(height=400)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Source distribution
        col1, col2 = st.columns(2)
        
        with col1:
            source_counts = df['source'].value_counts()
            fig_pie = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                title="Pollution Source Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Hourly pollution pattern
            hourly_pollution = df.groupby('hour')['pollution_index'].mean().reset_index()
            fig_hourly = px.bar(
                hourly_pollution,
                x='hour',
                y='pollution_index', 
                title="Average Pollution by Hour",
                color='pollution_index',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Interactive map
        st.header("🗺️ Interactive Pollution Map")
        
        # Create Folium map
        m = folium.Map(
            location=[latitude, longitude],
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        
        # Add source-colored markers
        source_colors = {
            'Vehicular': 'red',
            'Industrial': 'darkred',
            'Agricultural': 'green', 
            'Residential': 'blue'
        }
        
        for idx, row in df.sample(min(50, len(df))).iterrows():
            popup_text = f"""
            <b>Source:</b> {row['source']}<br>
            <b>PM2.5:</b> {row['PM2.5']:.1f} µg/m³<br>
            <b>Pollution Index:</b> {row['pollution_index']:.1f}<br>
            <b>Confidence:</b> {row['confidence']:.2f}
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=row['pollution_index'] / 10,
                popup=popup_text,
                color=source_colors.get(row['source'], 'gray'),
                fill=True,
                fillOpacity=0.7
            ).add_to(m)
        
        # Display map
        map_data = st_folium(m, width=700, height=500)
        
        # Recommendations section
        st.header("💡 Recommendations")
        
        recommendations = []
        if current_pollution > 150:
            recommendations.extend([
                "🏠 Stay indoors and use air purifiers",
                "😷 Wear N95 masks when going outside", 
                "🚫 Avoid outdoor exercise",
                "🌬️ Keep windows closed"
            ])
        elif current_pollution > 100:
            recommendations.extend([
                "⏰ Limit outdoor activities during peak hours",
                "😷 Consider wearing masks outdoors",
                "🏃‍♂️ Reduce intense outdoor exercise"
            ])
        else:
            recommendations.extend([
                "✅ Air quality is generally acceptable",
                "🏃‍♂️ Outdoor activities are safe for most people",
                "🌅 Consider exercising in early morning hours"
            ])
        
        for rec in recommendations:
            st.write(f"• {rec}")
        
        # Download section
        st.header("📥 Download Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📊 Download Data (CSV)",
                data=csv_data,
                file_name=f"pollution_data_{city}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Create summary report
            summary_report = f"""
            Pollution Analysis Report - {city}
            Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Current Status:
            - Pollution Index: {current_pollution:.1f}
            - Primary Source: {primary_source}
            - AQI Level: {get_aqi_level(current_pollution)[0]}
            
            Key Statistics:
            - Average PM2.5: {df['PM2.5'].mean():.1f} µg/m³
            - Average NO2: {df['NO2'].mean():.1f} µg/m³
            - Peak Pollution Hour: {df.groupby('hour')['pollution_index'].mean().idxmax()}
            - Most Common Source: {df['source'].mode()[0]}
            """
            
            st.download_button(
                label="📄 Download Report (TXT)",
                data=summary_report,
                file_name=f"pollution_report_{city}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

else:
    # Welcome screen
    st.info("👈 Please enter a city name and click 'Analyze Pollution' to begin the analysis.")
    
    # Sample visualizations
    st.subheader("🌟 What You'll Get:")
    st.write("• **Real-time pollution monitoring** with AQI levels")
    st.write("• **Source identification** using AI models")  
    st.write("• **Interactive maps** with pollution hotspots")
    st.write("• **Trend analysis** and predictions")
    st.write("• **Health recommendations** based on pollution levels")
    st.write("• **Downloadable reports** for further analysis")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🌍 EnviroScan - AI-Powered Pollution Source Identification System<br>"
    "Built with Streamlit • Data visualization with Plotly • Maps with Folium"
    "</div>", 
    unsafe_allow_html=True
)
'''
    
    return app_code

def create_dashboard_files():
    """
    Create all necessary files for the Streamlit dashboard
    """
    print("Creating Streamlit dashboard files...")
    
    # Create main app file
    app_code = create_streamlit_app()
    with open('/tmp/pollution_dashboard.py', 'w') as f:
        f.write(app_code)
    
    # Create requirements for Streamlit
    streamlit_requirements = """
streamlit==1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
folium>=0.14.0
streamlit-folium>=0.13.0
requests>=2.28.0
"""
    
    with open('/tmp/streamlit_requirements.txt', 'w') as f:
        f.write(streamlit_requirements)
    
    print("Dashboard files created in /tmp/")
    print("- pollution_dashboard.py: Main Streamlit application")
    print("- streamlit_requirements.txt: Required packages")
    
    return '/tmp/pollution_dashboard.py'


"""Main Workflow and Documentation"""

def run_complete_workflow(city="Delhi", save_model=True):
    """
    Execute the complete EnviroScan workflow
    
    Args:
        city (str): City name for analysis
        save_model (bool): Whether to save the trained model
        
    Returns:
        dict: Complete workflow results
    """
    print(f"🚀 Starting complete EnviroScan workflow for {city}")
    print("=" * 60)
    
    workflow_results = {}
    
    try:
        # Step 1: Data Collection
        print("\n📊 Step 1: Data Collection")
        raw_data = fetch_openaq_data(city)
        workflow_results['raw_data_shape'] = raw_data.shape
        print(f"✅ Collected {len(raw_data)} pollution measurements")
        
        # Step 2: Data Cleaning
        print("\n🧹 Step 2: Data Cleaning and Preprocessing")
        cleaned_data = clean_pollution_data(raw_data)
        workflow_results['cleaned_data_shape'] = cleaned_data.shape
        
        # Step 3: Feature Engineering  
        print("\n⚙️ Step 3: Feature Engineering")
        featured_data = feature_engineering(cleaned_data)
        workflow_results['featured_data_shape'] = featured_data.shape
        
        # Step 4: Source Labeling
        print("\n🏷️ Step 4: Source Labeling and Simulation")
        labeled_data = label_sources(featured_data)
        source_distribution = labeled_data['source'].value_counts()
        workflow_results['source_distribution'] = source_distribution.to_dict()
        print("Source distribution:")
        print(source_distribution)
        
        # Step 5: Dataset Splitting
        print("\n📊 Step 5: Dataset Splitting")
        train_df, val_df, test_df = split_dataset(labeled_data)
        workflow_results['dataset_splits'] = {
            'train': len(train_df),
            'validation': len(val_df), 
            'test': len(test_df)
        }
        
        # Step 6: Feature Preparation and Balancing
        print("\n⚖️ Step 6: Feature Preparation and Data Balancing")
        X_train, feature_names = prepare_features(train_df)
        y_train = train_df['source']
        
        X_val, _ = prepare_features(val_df)
        y_val = val_df['source']
        
        X_test, _ = prepare_features(test_df)
        y_test = test_df['source']
        
        # Apply SMOTE balancing
        X_train_balanced, y_train_balanced = apply_data_balancing(X_train, y_train)
        workflow_results['feature_names'] = feature_names
        
        # Step 7: Model Training
        print("\n🤖 Step 7: Model Training and Hyperparameter Tuning")
        trained_models = train_multiple_models(X_train_balanced, y_train_balanced, X_val, y_val)
        workflow_results['trained_models'] = list(trained_models.keys())
        
        # Step 8: Model Evaluation  
        print("\n📈 Step 8: Model Evaluation")
        performance_results = evaluate_model_performance(trained_models, X_test, y_test)
        workflow_results['model_performance'] = {
            model: {
                'accuracy': results['accuracy'],
                'f1_score': results['f1_score']
            }
            for model, results in performance_results.items()
        }
        
        # Step 9: Visualization Creation
        print("\n📊 Step 9: Performance Visualizations")
        create_performance_visualizations(performance_results, y_test)
        
        # Step 10: Model Saving
        if save_model:
            print("\n💾 Step 10: Model Saving")
            best_model_name = save_best_model(trained_models, performance_results, feature_names)
            workflow_results['best_model'] = best_model_name
        
        # Step 11: Predictions
        print("\n🔮 Step 11: Making Predictions")
        predictions_df = predict_pollution_sources(labeled_data)
        workflow_results['predictions_made'] = len(predictions_df)
        
        # Step 12: Geospatial Visualizations
        print("\n🗺️ Step 12: Creating Geospatial Visualizations")
        pollution_heatmap = create_pollution_heatmap(predictions_df)
        source_map = create_source_map(predictions_df)
        multi_layer_map = create_multi_layer_map(predictions_df)
        
        workflow_results['maps_created'] = [
            '/tmp/pollution_heatmap.html',
            '/tmp/source_map.html',
            '/tmp/multi_layer_map.html'
        ]
        
        # Step 13: Insights Generation
        print("\n💡 Step 13: Generating Insights")
        insights = generate_pollution_insights(predictions_df)
        workflow_results['insights'] = insights
        
        # Step 14: Dashboard Creation
        print("\n📱 Step 14: Creating Streamlit Dashboard")
        dashboard_path = create_dashboard_files()
        workflow_results['dashboard_path'] = dashboard_path
        
        print("\n✅ Complete workflow finished successfully!")
        print("=" * 60)
        
        # Print summary
        print("\n📋 WORKFLOW SUMMARY:")
        print(f"• Data processed: {workflow_results['raw_data_shape'][0]} → {workflow_results['cleaned_data_shape'][0]} samples")
        print(f"• Features engineered: {len(feature_names)} features")
        print(f"• Models trained: {len(trained_models)}")
        print(f"• Best model: {workflow_results.get('best_model', 'N/A')}")
        print(f"• Maps created: {len(workflow_results['maps_created'])}")
        print(f"• Dashboard: {dashboard_path}")
        
        return workflow_results
        
    except Exception as e:
        print(f"❌ Error in workflow: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

# Example usage and main execution
if __name__ == "__main__":
    # Run the complete workflow
    results = run_complete_workflow("Delhi")
    
    print("\n🎯 Next Steps:")
    print("1. Run the Streamlit dashboard: streamlit run /tmp/pollution_dashboard.py")
    print("2. View generated maps in /tmp/ directory")
    print("3. Check model performance visualizations")
    print("4. Deploy to production environment")
