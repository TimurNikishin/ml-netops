# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input # type: ignore
import argparse
import json

# Define command-line arguments
parser = argparse.ArgumentParser(description='Train a neural network to predict web app failures based on bandwidth data.')
parser.add_argument('--csv-path', type=str, required=True, help='Path to the CSV file containing training data.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training the model.')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training the model.')
parser.add_argument('--hidden-layers', type=int, nargs='+', default=[32, 16], help='List of neuron counts for each hidden layer (e.g., 64 32).')
parser.add_argument('--activation', type=str, default='relu', help='Activation function to use in hidden layers (e.g., relu, tanh).')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer.')

args = parser.parse_args()

# Load the CSV Data
df = pd.read_csv(args.csv_path)
df['web_app_failure'] = df['Web App Health (%)'].apply(lambda x: 1 if x < 95 else 0)
scaler = MinMaxScaler()
df['bandwidth_rate_normalized'] = scaler.fit_transform(df[['Bandwidth Rate (bps)']])

X = df[['bandwidth_rate_normalized']].values
Y = df['web_app_failure'].values

# Split the data into training (70%) and testing (30%) sets.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)

# Build the Neural Network Model
model = Sequential()
model.add(Input(shape=(1,)))

# Add hidden layers based on the provided argument
for neurons in args.hidden_layers:
    model.add(Dense(neurons, activation=args.activation))

model.add(Dense(1, activation='sigmoid'))

# Compile the Model
from tensorflow.keras.optimizers import Adam # type: ignore
model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model with minimal output
model.fit(X_train, Y_train, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.3, verbose=0)

# Evaluate the Model on Test Data with minimal output
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)

# Make Predictions on Test Data
predictions = (model.predict(X_test, verbose=0) > 0.5).astype("int32")

# Calculate classification metrics
report = classification_report(Y_test, predictions, output_dict=True)

# Extract precision, recall, and f1-score for class 0 and class 1
precision_0 = report['0']['precision']
recall_0 = report['0']['recall']
f1_0 = report['0']['f1-score']
precision_1 = report['1']['precision']
recall_1 = report['1']['recall']
f1_1 = report['1']['f1-score']

# Create a JSON object with all the metrics
result = {
    "accuracy": accuracy,
    "precision_0": precision_0,
    "recall_0": recall_0,
    "f1_0": f1_0,
    "precision_1": precision_1,
    "recall_1": recall_1,
    "f1_1": f1_1
}

# Print the JSON as a single line
print(json.dumps(result))
