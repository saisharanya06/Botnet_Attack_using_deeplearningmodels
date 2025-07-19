import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
import joblib

# Load dataset
data = pd.read_csv('C:\\Users\\saish\\OneDrive\\Desktop\\botnet\\preprocessed_unsw_nb15.csv')

# Separate features and target
X = data.drop('rate', axis=1).values
y = data['rate'].values  # Continuous regression target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the regression model
def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])  # Full loss function object
    return model

# Build and train
model = build_model()
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# ✅ Save model using new format
model.save('ann_model.keras')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
