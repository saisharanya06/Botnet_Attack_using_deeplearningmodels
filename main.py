import streamlit as st


import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import os
print("Current Working Directory:", os.getcwd())
print("Files in this folder:", os.listdir())


# Load models once (at start)
@st.cache_resource
def load_models_and_tools():
    ann_model = load_model("ann_model.h5")
    cnn_model = load_model("cnn_model.h5")
    rnn_model = load_model("rnn_model.h5")
    lstm_model = load_model("lstm_model.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    # Dummy accuracy and ROC data for example
    accuracy = {
        "ANN": 0.92,
        "CNN": 0.89,
        "RNN": 0.91,
        "LSTM": 0.90    
    }
    # Dummy ROC curve data (replace with real data)
    fpr = np.linspace(0, 1, 100)
    roc_data = {
        "ANN": (fpr, np.sqrt(fpr), 0.95),
        "CNN": (fpr, fpr**0.5, 0.93),
        "RNN": (fpr, 1-fpr**0.3, 0.94),
        "LSTM": (fpr, 1-fpr**0.4, 0.92),
    }

    return ann_model, cnn_model, rnn_model, lstm_model, scaler, label_encoder, accuracy, roc_data

ann_model, cnn_model, rnn_model, lstm_model, scaler, label_encoder, accuracy, roc_data = load_models_and_tools()

feature_names = ['sbytes', 'dbytes', 'rate', 'dinpkt', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean']

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "Visualizations", "About"])

if page == "Home":
    st.title("🛡️ Botnet Attack Detection")
    st.markdown("""
    Welcome to the Botnet Attack Detection app!  
    Navigate using the sidebar to predict attacks, see visualizations, or learn about the project.
    """)

elif page == "Predict":
    st.title("🔮 Predict Botnet Attack")

    # Input form
    with st.form("input_form"):
        st.write("Enter feature values:")
        inputs = {}
        for feature in feature_names:
            inputs[feature] = st.number_input(f"{feature}", format="%.5f")
        submitted = st.form_submit_button("Predict")

    if submitted:
        user_array = np.array([list(inputs.values())]).astype(np.float32)
        user_scaled = scaler.transform(user_array)

        # Predict probabilities from each model
        ann_pred = ann_model.predict(user_scaled)[0]
        cnn_pred = cnn_model.predict(user_scaled)[0]
        rnn_pred = rnn_model.predict(user_scaled)[0]
        lstm_pred = lstm_model.predict(user_scaled)[0]

        # Labels (argmax)
        ann_label = np.argmax(ann_pred)
        cnn_label = np.argmax(cnn_pred)
        rnn_label = np.argmax(rnn_pred)
        lstm_label = np.argmax(lstm_pred)

        votes = [ann_label, cnn_label, rnn_label, lstm_label]
        final_label = max(set(votes), key=votes.count)
        attack_name = label_encoder.inverse_transform([final_label])[0]
        binary_label = 0 if attack_name.lower() == "normal" else 1

        st.subheader("Prediction Results")
        st.write(f"**Attack Type Detected:** {attack_name}")
        st.write(f"**Binary Label:** {binary_label} ({'Normal' if binary_label == 0 else 'Attack'})")

        # Show model prediction probabilities as a bar chart
        st.subheader("Model Prediction Probabilities")
        probs_df = {
            "ANN": ann_pred,
            "CNN": cnn_pred,
            "RNN": rnn_pred,
            "LSTM": lstm_pred,
        }
        import pandas as pd
        df = pd.DataFrame(probs_df, index=label_encoder.classes_).T
        st.bar_chart(df)

elif page == "Visualizations":
    st.title("📊 Model Performance Visualizations")

    # Accuracy comparison bar chart
    st.subheader("Model Accuracy Comparison")
    import pandas as pd
    acc_df = pd.DataFrame.from_dict(accuracy, orient="index", columns=["Accuracy"])
    st.bar_chart(acc_df)

    # ROC Curves
    st.subheader("ROC Curves for All Models")
    plt.figure(figsize=(8,6))
    for model_name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0,1],[0,1],'k--', lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    st.pyplot(plt)

    

elif page == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    **Project:** Botnet Attack Detection using Deep Learning  
    **Dataset:** UNSW-NB15 Dataset  
    **Models Used:** ANN, CNN, RNN, LSTM  
    \n
    This project detects botnet attacks based on network traffic features using an ensemble of four deep learning models.  
    The app supports multi-page navigation, model prediction, performance visualization, and an informative about page.  
    \n
    
    """)

