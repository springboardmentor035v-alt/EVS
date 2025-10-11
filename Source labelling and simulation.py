import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# --- Source Labeling & Simulation ---
def label_sources(df):
    # Start with all Unknown
    df['source'] = 'Unknown'

    # Vehicular source: near road + high NO2
    df.loc[(df['roads_count'] > 50) & (df['pollution_count'] > 3), 'source'] = 'Vehicular'

    # Industrial source: near factories + sulfur dioxide / high infra_score
    df.loc[(df['factories_count'] > 5) & (df['infra_score'] > 2.0), 'source'] = 'Industrial'

    # Agricultural source: rural-like areas + high PM (simulate using temp/humidity as proxy)
    df.loc[(df['roads_count'] < 20) & (df['weather_category'] == 'Dry') & (df['pollution_count'] > 5),
           'source'] = 'Agricultural'

    # Simulated labeling: if still Unknown, assign probabilistically
    np.random.seed(42)
    df.loc[df['source'] == 'Unknown', 'source'] = np.random.choice(
        ['Vehicular', 'Industrial', 'Agricultural'],
        size=(df['source'] == 'Unknown').sum(),
        p=[0.4, 0.3, 0.3]  # probabilities
    )

    return df


# --- Train/Validation/Test Split ---
def split_dataset(df):
    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['source'])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['source'])
    return train, val, test


# --- Balancing with SMOTE (if needed) ---
def balance_data(train):
    # Exclude non-numeric columns before applying SMOTE
    X = train.drop(columns=['source', 'city', 'weather_condition', 'weather_category'])
    y = train['source']

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = smote.fit_resample(X, y)

    train_balanced = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.DataFrame(y_res, columns=['source'])], axis=1)
    return train_balanced


# --- Pipeline Execution ---
df = pd.read_csv("collected_data_cleaned.csv")

# Label sources
df_labeled = label_sources(df)

# Split into sets
train, val, test = split_dataset(df_labeled)

# Balance training data
train_balanced = balance_data(train)

# Save all
train_balanced.to_csv("train_balanced.csv", index=False)
val.to_csv("validation.csv", index=False)
test.to_csv("test.csv", index=False)

print("✅ Labeled, split, and balanced data saved!")
