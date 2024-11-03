import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model("network_failure_model.keras")

# Load the maximum bandwidth from max_bandwidth.txt
with open("max_bandwidth.txt", "r") as f:
    max_bandwidth = float(f.readline().strip())

# Load new data from a CSV file
df = pd.read_csv("data-collection\\training-data\\data_set_for_failure_prediction.csv")

# Ensure "Timestamp" column is in datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Calculate time intervals in hours from the first timestamp
df['Time (hours)'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() / 3600

# Normalize bandwidth using the max_bandwidth from the file
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

# Plot actual bandwidth data, trend line, and future projection
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot bandwidth rate over time on primary Y-axis
ax1.plot(df['Time (hours)'], Y_bandwidth, label="Actual Bandwidth Rate (bps)", color='blue')
ax1.plot(df['Time (hours)'], trend_line, label="Bandwidth Trend Line", linestyle="--", color='orange')
ax1.plot(future_time, future_bandwidth, label="Projected Future Bandwidth", linestyle="--", color='green')
ax1.set_xlabel("Time (Hours)")
ax1.set_ylabel("Bandwidth Rate (bps)")
ax1.set_title("Bandwidth Rate and Projected Trend Over Time")
ax1.legend(loc="upper left")
ax1.grid(True)

# Load the training dataset to get min and max bandwidth values
df_ref = pd.read_csv("data-collection/training-data/combined_output_2024-10-28_19-08-22.csv")
scaler = MinMaxScaler()
df_ref['bandwidth_rate_normalized'] = scaler.fit_transform(df_ref[['Bandwidth Rate (bps)']])

# Set y-axis ticks for bandwidth rate based on training dataset's min and max bandwidth
min_bandwidth = df_ref['Bandwidth Rate (bps)'].min()
max_bandwidth = df_ref['Bandwidth Rate (bps)'].max()
ax1.set_yticks(np.linspace(min_bandwidth, max_bandwidth, 10))
ax1.set_yticklabels([f"{int(y):,}" for y in np.linspace(min_bandwidth, max_bandwidth, 10)])

# Generate bandwidth rates for failure prediction
bandwidth_rates_actual = np.linspace(min_bandwidth, max_bandwidth, 100).reshape(-1, 1)
normalized_bandwidth_rates = scaler.transform(bandwidth_rates_actual)
failure_probabilities = model.predict(normalized_bandwidth_rates, verbose=0).flatten() * 100  # Convert to percentage

# Calculate failure probability thresholds
thresholds = np.arange(10, 101, 10)
bandwidth_at_thresholds = {}

for threshold in thresholds:
    idx = (np.abs(failure_probabilities - threshold)).argmin()
    bandwidth_at_thresholds[threshold] = bandwidth_rates_actual[idx][0]

# Add failure probability thresholds on secondary Y-axis without labels
ax2 = ax1.twinx()
ax2.set_ylabel("Failure Probability (%)")

# Set y-axis ticks for failure probability but hide labels
ax2.set_yticks(list(bandwidth_at_thresholds.values()))
ax2.set_yticklabels([])  # Hide the labels for the probability Y-axis

# Plot threshold lines and add indented labels on the left side of bandwidth Y-axis
for i, (prob, bandwidth) in enumerate(bandwidth_at_thresholds.items()):
    x_position = last_time * 0.05  # Set position to the left side with indentation
    indent = i * 4  # Increase indentation for each subsequent label
    ax1.axhline(y=bandwidth, linestyle="--", color="red", linewidth=0.5)
    ax1.text(x_position + indent, bandwidth, f"{prob}%", color="red",
             va="center", ha="left", fontweight="bold")

# Set y-axis to start from 0 to avoid negative values
ax1.set_ylim(bottom=0)

plt.tight_layout()
plt.show()
