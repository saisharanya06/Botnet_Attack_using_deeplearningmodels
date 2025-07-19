import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load preprocessed dataset
data = pd.read_csv('preprocessed_unsw_nb15.csv')

# Load label encoder and scaler
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare features and labels
X = data.drop('attack_cat', axis=1).values
y = label_encoder.transform(data['attack_cat'])
y_categorical = to_categorical(y)

# Scale features
X_scaled = scaler.transform(X)

# Split into train/test (same random_state used in training)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Load models
ann_model = load_model('ann_model.h5')
cnn_model = load_model('cnn_model.h5')
rnn_model = load_model('rnn_model.h5')
lstm_model = load_model('lstm_model.h5')

# Evaluate models
ann_loss, ann_accuracy = ann_model.evaluate(X_test, y_test, verbose=0)
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)
rnn_loss, rnn_accuracy = rnn_model.evaluate(X_test, y_test, verbose=0)
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test, y_test, verbose=0)

# Print results
print("✅ Model Accuracy Results")
print(f"🔸 ANN Accuracy  : {ann_accuracy:.4f}")
print(f"🔸 CNN Accuracy  : {cnn_accuracy:.4f}")
print(f"🔸 RNN Accuracy  : {rnn_accuracy:.4f}")
print(f"🔸 LSTM Accuracy : {lstm_accuracy:.4f}")
