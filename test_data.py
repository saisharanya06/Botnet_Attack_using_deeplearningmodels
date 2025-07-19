import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load preprocessed data
data = pd.read_csv("preprocessed_unsw_nb15.csv")

# Define features and label
feature_columns = ['sbytes', 'dbytes', 'rate', 'dinpkt', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean']
X = data[feature_columns]
y = data['attack_cat']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler and label encoder (in case you didn't already)
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Save test data
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")

print("✅ Done: Saved X_test.pkl and y_test.pkl for ROC curves.")
