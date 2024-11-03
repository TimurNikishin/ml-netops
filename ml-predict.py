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

# Plot only the actual bandwidth data and the trend line
plt.figure(figsize=(10, 6))
plt.plot(df['Time (hours)'], Y_bandwidth, label="Actual Bandwidth Rate (bps)", color='blue')
plt.plot(df['Time (hours)'], trend_line, label="Bandwidth Trend Line", linestyle="--", color='orange')
plt.plot(future_time, future_bandwidth, label="Projected Future Bandwidth", linestyle="--", color='green')
plt.xlabel("Time (Hours)")
plt.ylabel("Bandwidth Rate (bps)")
plt.title("Bandwidth Rate and Projected Trend Over Time")
plt.legend()
plt.grid(True)

# Set y-axis to start from 0 to avoid negative values
plt.ylim(bottom=0)

# Adjust y-axis tick labels to display bandwidth in bps format
plt.yticks([int(y) for y in plt.yticks()[0]], [f"{int(y):,}" for y in plt.yticks()[0]])

plt.tight_layout()
plt.show()
