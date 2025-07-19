import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# 👇 Update this to your actual CSV file name
df = pd.read_csv("C:\\Users\\saish\\OneDrive\\Desktop\\botnet\\preprocessed_unsw_nb15.csv")

# 👇 Only include these 9 features
feature_columns = ['sbytes', 'dbytes', 'rate', 'dinpkt', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean']
X = df[feature_columns]

# 🔄 Retrain scaler
scaler = StandardScaler()
scaler.fit(X)

# 💾 Save the updated scaler
joblib.dump(scaler, 'scaler.pkl')

print("✅ Scaler retrained and saved with 9 features.")
