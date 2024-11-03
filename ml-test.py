import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model # type: ignore


# Load the trained model
model = load_model('network_failure_model.keras')

# Load the CSV data (used for training)
# Replace with the actual path to your CSV file that was used for training
df = pd.read_csv('data-collection\\training-data\\combined_output_2024-10-28_19-08-22.csv')

# Example bandwidth rate (in bps) to test manually
example_bandwidth_rate = 15000000  # Replace with the bandwidth rate you want to test (in bps)

# Initialize the MinMaxScaler and fit it using the training data's bandwidth rates
train_bandwidth_data = df[['Bandwidth Rate (bps)']]  # Use the training data for scaling
scaler = MinMaxScaler()
scaler.fit(train_bandwidth_data)

# Normalize the input value using the same scaler
normalized_rate = scaler.transform([[example_bandwidth_rate]])

# Make a prediction using the model (output will be a probability)
prediction = model.predict(normalized_rate)

# Interpret the prediction (threshold = 0.5 for binary classification)
if prediction > 0.5:
    print(f"Prediction: Web app will fail. (Predicted probability: {prediction[0][0]:.4f})")
else:
    print(f"Prediction: Web app will NOT fail. (Predicted probability: {prediction[0][0]:.4f})")
