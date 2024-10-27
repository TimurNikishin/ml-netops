# Import necessary libraries
import pandas as pd                                        # For loading and manipulating data (from CSV)
import numpy as np                                         # For numerical operations
from sklearn.model_selection import train_test_split       # For splitting data into training and testing sets
from sklearn.preprocessing import MinMaxScaler             # For scaling features to a specific range
from sklearn.metrics import classification_report          # For evaluating model performance (precision, recall, f1-score)
from tensorflow.keras.models import Sequential             # type: ignore # For building a sequential neural network
from tensorflow.keras.layers import Dense, Input           # type: ignore # For creating input and dense layers in the neural network



# 1. Load the CSV Data
# Replace 'your_data.csv' with the path to your actual CSV file containing the training data
df = pd.read_csv('data-collection\\training-data\\combined_output_2024-10-27_11-13-36.csv')

# 2. Preprocess the Data

# Create a binary label for web app failure based on web app health percentage.
# If 'Web App Health (%)' is less than 95, it's considered a failure (1), otherwise it's healthy (0).
df['web_app_failure'] = df['Web App Health (%)'].apply(lambda x: 1 if x < 95 else 0)

# Normalize the 'Bandwidth Rate (bps)' to a range between 0 and 1 using MinMaxScaler
# Scaling helps the neural network to process the values more effectively.
scaler = MinMaxScaler()
df['bandwidth_rate_normalized'] = scaler.fit_transform(df[['Bandwidth Rate (bps)']])

# Define the input features (X) and target labels (Y)
# X: Normalized bandwidth rates
# Y: Binary labels for web app failure
X = df[['bandwidth_rate_normalized']].values  # Input feature (bandwidth rate)
Y = df['web_app_failure'].values              # Target label (web app failure or no failure)


# Split the data into training (70%) and testing (30%) sets.
# shuffle=False ensures that the time sequence is maintained (as it's time-series data).
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)

# 3. Build the Neural Network Model

# Initialize the Sequential model
model = Sequential()

# Add the input layer with 1 input feature (bandwidth rate)
model.add(Input(shape=(1,)))  # Input shape is (1,) because we have only 1 feature (normalized bandwidth rate)

# Add the first hidden layer with 64 neurons and 'relu' activation function
model.add(Dense(32, activation='relu'))  # ReLU activation helps introduce non-linearity to the model

# Add the second hidden layer with 32 neurons and 'relu' activation function
model.add(Dense(16, activation='relu'))  # Smaller number of neurons in subsequent layers is common practice

# Add the output layer with 1 neuron and 'sigmoid' activation function
# The sigmoid function outputs a value between 0 and 1, which is used for binary classification.
model.add(Dense(1, activation='sigmoid'))

# 4. Compile the Model
# Compile the model using the 'adam' optimizer, and 'binary_crossentropy' loss function for binary classification.
# The 'accuracy' metric is used to track performance during training and evaluation.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train the Model
# Train the model on the training data (X_train, Y_train)
# Use 50 epochs (iterations over the entire dataset) and a batch size of 32 (samples per gradient update)
# Also, reserve 30% of the training data for validation to monitor the model’s performance during training.
model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.3)

# 6. Evaluate the Model on Test Data
# Evaluate the model's performance on the unseen test data (X_test, Y_test)
# The evaluate method returns the loss and accuracy on the test set.
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {accuracy}")

# 7. Make Predictions on Test Data
# Predict binary outcomes (failure or no failure) for the test data.
# The model outputs a probability between 0 and 1. We use a threshold of 0.5 to classify it as failure (1) or no failure (0).
predictions = (model.predict(X_test) > 0.5).astype("int32")  # Convert predictions to 0 or 1

# Display classification metrics such as precision, recall, and F1-score for each class (failure or no failure).
print(classification_report(Y_test, predictions))

# 8. Save the Trained Model for Future Use
# Save the model in Keras's recommended format (.keras) for future use
model.save('network_failure_model.keras')


# Optional: To load the model later, use the following code
# from tensorflow.keras.models import load_model
# loaded_model = load_model('network_failure_model.keras')
