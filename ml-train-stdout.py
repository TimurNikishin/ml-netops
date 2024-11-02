# Import necessary libraries
import pandas as pd                                        # For loading and manipulating data (from CSV)
import numpy as np                                         # For numerical operations
from sklearn.model_selection import train_test_split       # For splitting data into training and testing sets
from sklearn.preprocessing import MinMaxScaler             # For scaling features to a specific range
from sklearn.metrics import classification_report          # For evaluating model performance (precision, recall, f1-score)
from tensorflow.keras.models import Sequential             # type: ignore # For building a sequential neural network
from tensorflow.keras.layers import Dense, Input           # type: ignore # For creating input and dense layers in the neural network
import argparse                                            # For parsing command-line arguments

# 1. Define command-line arguments
parser = argparse.ArgumentParser(description='Train a neural network to predict web app failures based on bandwidth data.')

parser.add_argument('--csv-path', type=str, required=True, help='Path to the CSV file containing training data.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training the model.')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training the model.')
parser.add_argument('--hidden-layers', type=int, nargs='+', default=[32, 16], help='List of neuron counts for each hidden layer (e.g., 64 32).')
parser.add_argument('--activation', type=str, default='relu', help='Activation function to use in hidden layers (e.g., relu, tanh).')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer.')

args = parser.parse_args()

# 2. Load the CSV Data
df = pd.read_csv(args.csv_path)

# Preprocess the Data
df['web_app_failure'] = df['Web App Health (%)'].apply(lambda x: 1 if x < 95 else 0)

scaler = MinMaxScaler()
df['bandwidth_rate_normalized'] = scaler.fit_transform(df[['Bandwidth Rate (bps)']])

# Save the maximum bandwidth value for use in prediction
max_bandwidth = df['Bandwidth Rate (bps)'].max()
with open('max_bandwidth.txt', 'w') as f:
    f.write(str(max_bandwidth))

X = df[['bandwidth_rate_normalized']].values
Y = df['web_app_failure'].values

# Split the data into training (70%) and testing (30%) sets.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)

# 3. Build the Neural Network Model
model = Sequential()

# Add the input layer
model.add(Input(shape=(1,)))

# Add hidden layers based on the provided argument
for neurons in args.hidden_layers:
    model.add(Dense(neurons, activation=args.activation))

# Add the output layer
model.add(Dense(1, activation='sigmoid'))

# 4. Compile the Model
from tensorflow.keras.optimizers import Adam # type: ignore
model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train the Model
model.fit(X_train, Y_train, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.3)

# 6. Evaluate the Model on Test Data
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {accuracy}")

# 7. Make Predictions on Test Data
predictions = (model.predict(X_test) > 0.5).astype("int32")

# Display classification metrics such as precision, recall, and F1-score.
print(classification_report(Y_test, predictions))

# 8. Save the Trained Model
model.save('network_failure_model.keras')

