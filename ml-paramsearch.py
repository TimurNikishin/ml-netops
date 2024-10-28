import itertools
import subprocess
import pandas as pd
import sys
import os
import json

# Define possible values for each hyperparameter
csv_path = 'data-collection/training-data/combined_output_2024-10-27_11-13-36.csv'
epochs = [2, 5, 10, 25, 50, 100]  # Number of epochs to test
#batch_sizes = [32]  # Batch sizes to test
batch_sizes = [32, 64]  # Batch sizes to test
#hidden_layers_options = [[32, 16]]  # Hidden layer configurations
hidden_layers_options = [[16, 8], [32, 16], [64, 32]]  # Hidden layer configurations
#activation_functions = ['relu']  # Activation functions to test
activation_functions = ['relu', 'tanh']  # Activation functions to test
#earning_rates = [0.001]  # Learning rates to test
learning_rates = [0.001, 0.0001]  # Learning rates to test


# Get the path of the current Python interpreter
python_interpreter = sys.executable

# Prepare the environment for subprocess
env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'  # Ensure UTF-8 encoding

# List to store results
results = []

# Loop through all combinations of hyperparameters
for epochs_val, batch_size, hidden_layers, activation, lr in itertools.product(
    epochs, batch_sizes, hidden_layers_options, activation_functions, learning_rates):
    
    # Construct the command for running ml-train.py with the current hyperparameter combination
    command = [
        python_interpreter, 'ml-train.py',
        '--csv-path', csv_path,
        '--epochs', str(epochs_val),
        '--batch-size', str(batch_size),
        '--hidden-layers', *map(str, hidden_layers),
        '--activation', activation,
        '--learning-rate', str(lr)
    ]
    
    print(f"Running: {' '.join(command)}")
    
    # Run the command and capture the output, using errors='ignore' to bypass decoding issues
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            encoding='utf-8'
        )
        
        # Check if the process completed successfully
        if process.returncode != 0:
            print(f"Error running the command: {' '.join(command)}")
            print(f"Return code: {process.returncode}")
            print(f"Standard Error:\n{process.stderr}")
            continue  # Skip this iteration and move on to the next one

        # Parse the JSON output
        stdout_output = process.stdout.strip()
        print(f"JSON Output: {stdout_output}")  # Print the raw JSON for debugging

        metrics = json.loads(stdout_output)

        # Save the results for this configuration
        results.append({
            'epochs': epochs_val,
            'batch_size': batch_size,
            'hidden_layers': hidden_layers,
            'activation': activation,
            'learning_rate': lr,
            'accuracy': metrics["accuracy"],
            'precision_0': metrics["precision_0"],
            'recall_0': metrics["recall_0"],
            'f1_0': metrics["f1_0"],
            'precision_1': metrics["precision_1"],
            'recall_1': metrics["recall_1"],
            'f1_1': metrics["f1_1"]
        })

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}. Output was:\n{stdout_output}")
        continue
    except Exception as e:
        print(f"Unexpected error: {e}")
        continue

# Create a DataFrame to store all the results
results_df = pd.DataFrame(results)

# Sort the results by accuracy in descending order to find the best settings
if not results_df.empty:
    results_df = results_df.sort_values(by='accuracy', ascending=False)

    # Save the results to a CSV file
    results_df.to_csv('parameter_search_results.csv', index=False)
    print("Parameter search completed. Results saved to 'parameter_search_results.csv'.")
else:
    print("No valid results to save.")
