import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt # type: ignore

# Load the trained model
model = load_model('network_failure_model.keras')

# Load the dataset
df = pd.read_csv('data-collection/training-data/combined_output_2024-10-27_11-13-36.csv')

# Add the 'web_app_failure' column if it doesn't exist
if 'web_app_failure' not in df.columns:
    df['web_app_failure'] = df['Web App Health (%)'].apply(lambda x: 1 if x < 95 else 0)

# Check actual min and max bandwidth rates
min_bandwidth = df['Bandwidth Rate (bps)'].min()
max_bandwidth = df['Bandwidth Rate (bps)'].max()

# Generate bandwidth rates within the actual range
bandwidth_rates_actual = np.linspace(min_bandwidth, max_bandwidth, 100).reshape(-1, 1)

# Scale these bandwidth rates for model input
scaler = MinMaxScaler()
df['bandwidth_rate_normalized'] = scaler.fit_transform(df[['Bandwidth Rate (bps)']])
normalized_bandwidth_rates = scaler.transform(bandwidth_rates_actual)

# Predict failure probability using the scaled values
failure_probabilities = model.predict(normalized_bandwidth_rates, verbose=0).flatten() * 100  # Convert to percentage

# Find bandwidth values for specific failure probabilities (10%, 20%, ..., 100%)
thresholds = np.arange(10, 101, 10)
bandwidth_at_thresholds = {}

for threshold in thresholds:
    # Find the closest point in failure probabilities to the threshold
    idx = (np.abs(failure_probabilities - threshold)).argmin()
    bandwidth_at_thresholds[threshold] = bandwidth_rates_actual[idx][0]

# Print out the bandwidth values for each threshold
print("Predictions for application failure at different bandwidth rates:")
for threshold, bandwidth in bandwidth_at_thresholds.items():
    print(f"At a bandwidth rate of {int(bandwidth):,} bps, the application will fail with a {threshold}% probability.")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(bandwidth_rates_actual.flatten(), failure_probabilities, label="Predicted Failure Probability", color="blue")
plt.xlabel("Bandwidth Rate (bps)")
plt.ylabel("Failure Probability (%)")
plt.title("Web Application Failure Probability vs. Bandwidth Rate")
plt.grid(True)
plt.legend()

# Set custom x-axis tick labels to show actual bandwidth values in bps
plt.xticks(np.linspace(min_bandwidth, max_bandwidth, 10), [f"{int(x):,}" for x in np.linspace(min_bandwidth, max_bandwidth, 10)], rotation=45)

plt.tight_layout()
plt.show()
