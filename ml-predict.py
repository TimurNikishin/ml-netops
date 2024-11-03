import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model("network_failure_model.keras")

# Load the maximum bandwidth from max_bandwidth.txt
with open("max_bandwidth.txt", "r") as f:
    max_bandwidth = float(f.readline().strip())

# Load new data from a CSV file (live data for trend analysis)
df = pd.read_csv("data-collection\\training-data\\data_set_for_failure_prediction.csv")

# Ensure "Timestamp" column is in datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Calculate time intervals in hours from the first timestamp
df['Time (hours)'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() / 3600

# Normalize the live data bandwidth using the max_bandwidth from the file
df['normalized_bandwidth'] = df['Bandwidth Rate (bps)'] / max_bandwidth

# Fit a linear regression model to show bandwidth trend over time
X_time = df['Time (hours)'].values.reshape(-1, 1)  # Use calculated time in hours as x-axis
Y_bandwidth = df['Bandwidth Rate (bps)'].values
regression_model = LinearRegression()
regression_model.fit(X_time, Y_bandwidth)
trend_line = regression_model.predict(X_time)

# Project future bandwidth growth using the linear regression model
future_time_steps = 50  # Number of time steps to predict into the future
last_time = df['Time (hours)'].iloc[-1]  # Get the last time in hours
future_time = np.arange(last_time + 1, last_time + 1 + future_time_steps).reshape(-1, 1)
future_bandwidth = regression_model.predict(future_time)

# --- Calculate Failure Probability Thresholds from Training Data ---

# Load training data for calculating probability thresholds
training_df = pd.read_csv("data-collection/training-data/combined_output_2024-10-28_19-08-22.csv")

# Calculate min and max bandwidth for normalization based on training data
scaler = MinMaxScaler()
training_df['normalized_bandwidth'] = scaler.fit_transform(training_df[['Bandwidth Rate (bps)']])

# Generate a range of bandwidth rates for failure probability calculation
bandwidth_range = np.linspace(training_df['Bandwidth Rate (bps)'].min(), training_df['Bandwidth Rate (bps)'].max(), 100).reshape(-1, 1)
normalized_bandwidth_range = scaler.transform(bandwidth_range)

# Predict failure probabilities
failure_probabilities = model.predict(normalized_bandwidth_range).flatten() * 100

# Calculate bandwidth thresholds for each failure probability level (10%, 20%, ..., 100%)
thresholds = np.arange(10, 101, 10)
bandwidth_thresholds = {}

for threshold in thresholds:
    idx = (np.abs(failure_probabilities - threshold)).argmin()
    bandwidth_thresholds[threshold] = bandwidth_range[idx][0]

# --- Plotting ---

plt.figure(figsize=(10, 6))

# Plot actual bandwidth data and the trend line from live data
plt.plot(df['Time (hours)'], Y_bandwidth, label="Actual Bandwidth Rate (bps)", color='blue')
plt.plot(df['Time (hours)'], trend_line, label="Bandwidth Trend Line", linestyle="--", color='orange')
plt.plot(future_time, future_bandwidth, label="Projected Future Bandwidth", linestyle="--", color='green')

# Add failure probability thresholds as horizontal lines
for threshold, bw in bandwidth_thresholds.items():
    plt.axhline(y=bw, color='red', linestyle=":", label=f"{threshold}% Failure Probability at {int(bw):,} bps")

plt.xlabel("Time (Hours)")
plt.ylabel("Bandwidth Rate (bps)")
plt.title("Bandwidth Rate and Projected Trend with Failure Probability Thresholds")
plt.legend()
plt.grid(True)

# Set y-axis to start from 0 to avoid negative values
plt.ylim(bottom=0)

# Adjust y-axis tick labels to display bandwidth in bps format
plt.yticks([int(y) for y in plt.yticks()[0]], [f"{int(y):,}" for y in plt.yticks()[0]])

plt.tight_layout()
plt.show()
