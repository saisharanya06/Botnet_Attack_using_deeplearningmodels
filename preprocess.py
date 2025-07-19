import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
import joblib

# Load preprocessed dataset
data = pd.read_csv('C:\\Users\\saish\\OneDrive\\Desktop\\botnet\\preprocessed_unsw_nb15.csv')

# Drop ID column if exists
if 'id' in data.columns:
    data = data.drop('id', axis=1)

# Mapping from rate values to attack names
rate_to_attack = {
    111111.1072: 'Backdoor',
    125000.0003: 'Analysis',
    142857.1409: 'DoS',
    333333.3215: 'Generic',
    73.12267: 'Exploits',
    34.050365: 'Fuzzers',
    74.08749: 'Normal',
    83.601785: 'Reconnaissance',
    76923.07779: 'Shellcode',
    233.370165: 'Worms'
}

# Round rate values for matching
data['rate'] = data['rate'].round(5)

# Map rate values to attack names
data['attack_cat'] = data['rate'].map(rate_to_attack)

# Drop rows with unmapped rate values
data = data.dropna(subset=['attack_cat'])

# Save attack name list
attack_names = sorted(data['attack_cat'].unique())
joblib.dump(attack_names, 'attack_names.pkl')

# Encode attack labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['attack_cat'])
y_categorical = to_categorical(y)

# Save the label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# Features (drop rate and attack_cat)
X = data.drop(['rate', 'attack_cat'], axis=1).values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# Save processed data
np.savez('botnet_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

print("✅ Preprocessing complete. Data saved.")
