Botnet Detection Using Deep Learning 
This project is focused on detecting botnet attacks in network traffic using deep learning techniques. It leverages various neural network architectures to identify malicious behaviors and classify attack types effectively.

Project Overview Botnets are one of the most dangerous threats in cybersecurity. They hijack networked devices to perform attacks like DDoS, spamming, or data theft. Early and accurate detection is critical. In this project, we built multiple deep learning models using the UNSW-NB15 dataset, a comprehensive dataset that includes real-world normal and attack traffic. The goal is to predict the type of attack based on network traffic features.

Models Implemented We trained and compared the performance of the following deep learning models: 
 Artificial Neural Network (ANN),Convolutional Neural Network (CNN),Recurrent Neural Network (RNN),Long Short-Term Memory (LSTM)
Each model was evaluated using: Accuracy Confusion Matrix Classification Report (Precision, Recall, F1-Score) ROC Curves

Features Used Key features used for prediction: 
sbytes, dbytes, rate, dinpkt, tcprtt, synack, ackdat, smean, dmean These were normalized and split into training/testing sets after preprocessing.

Visualizations:
 Model Accuracy Comparison Chart,Confusion Matrix Heatmap,ROC Curves for Each Model,Feature Importance (from Random Forest)

Final Output Predicts the attack type (e.g., DoS, Exploits, Reconnaissance, etc.) Also outputs binary classification: 0 = Normal, 1 = Attack

Technologies Used Python TensorFlow / Keras Scikit-learn Matplotlib & Seaborn Pandas & NumPy
GUI (Optional) We also built a Tkinter/Streamlit-based GUI that allows users to input network parameters and view: The predicted attack type Accuracy comparison ROC curves per model
Future Work Real-time detection with packet sniffing tools Ensemble learning for even higher accuracy Deployment via a web app or API

## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/Botnet_Attack_using_deeplearningmodels.git
   cd Botnet_Attack_using_deeplearningmodels
