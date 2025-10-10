import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data():
    df = pd.read_csv("data/raw_data.csv")

    # Handle missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Normalize numerical columns
    cols_to_scale = ["PM25", "NO2", "Wind_Speed", "Temp"]
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # Add simple rule-based labels (for training)
    df["Pollution_Source"] = df.apply(
        lambda x: "Industrial" if x["NO2"] > 0.6
        else "Vehicular" if x["PM25"] > 0.7
        else "Natural",
        axis=1
    )

    df.to_csv("data/labeled_predictions.csv", index=False)
    print("✅ Cleaned and labeled data saved as data/labeled_predictions.csv")

if __name__ == "__main__":
    preprocess_data()
