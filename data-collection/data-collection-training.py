import os
import requests
import csv
from datetime import datetime, timedelta, timezone
import argparse
import sys

# Define the folder where you want to save the output
output_folder = "training-data"  

# Check if the folder exists, and if not, create it
os.makedirs(output_folder, exist_ok=True)

# Function to generate Prometheus query
def generate_query_range(prometheus_url, query, start_time, end_time, step):
    return f"{prometheus_url}/api/v1/query_range?query={query}&start={start_time}&end={end_time}&step={step}"

# Function to convert a date to a UNIX timestamp
def date_to_unix_timestamp(days_ago):
    target_date = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return int(target_date.timestamp())

# Function to get the bandwidth data from Prometheus
def get_bandwidth_data(prometheus_url, router_ip, interface_name, days, step):
    start_time = date_to_unix_timestamp(days)
    end_time = int(datetime.now(timezone.utc).timestamp())  # Now
    query = f'rate(ifHCInOctets{{ifName="{interface_name}", instance="{router_ip}:161"}}[30s]) * 8'
    
    url = generate_query_range(prometheus_url, query, start_time, end_time, step)
    
    response = requests.get(url)
    response_json = response.json()
    
    if response_json['status'] != 'success':
        print(f"Error querying bandwidth data: {response_json['error']}")
        sys.exit(1)
    
    return response_json['data']['result'][0]['values']  # List of [timestamp, bandwidth_rate]

# Function to get the web app health from Prometheus
def get_web_app_health(prometheus_url, web_app_url, days, step):
    start_time = date_to_unix_timestamp(days)
    end_time = int(datetime.now(timezone.utc).timestamp())  # Now
    query = f'avg_over_time(probe_success{{instance="{web_app_url}"}}[30s])'
    
    url = generate_query_range(prometheus_url, query, start_time, end_time, step)
    
    response = requests.get(url)
    response_json = response.json()
    
    if response_json['status'] != 'success':
        print(f"Error querying web app health data: {response_json['error']}")
        sys.exit(1)
    
    # Check if the result array is empty
    if not response_json['data']['result']:
        print(f"No data returned for web app instance {web_app_url}. Please check if the instance exists and has data in the specified time range.")
        sys.exit(1)
    
    return response_json['data']['result'][0]['values']  # List of [timestamp, web_app_health_percentage]

# Function to combine bandwidth and web app health data
def combine_data(bandwidth_data, web_app_health_data):
    combined_data = []
    
    # Assuming both datasets have the same timestamps, we can just zip them together
    for (bw_timestamp, bw_value), (health_timestamp, health_value) in zip(bandwidth_data, web_app_health_data):
        # Use timezone-aware datetime conversion
        combined_data.append([
            datetime.fromtimestamp(float(bw_timestamp), timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            int(float(bw_value)),  # Bandwidth in integer (bps)
            round(float(health_value) * 100, 2)  # Web app health percentage with two decimals
        ])
    
    return combined_data

# Function to write the combined data to a CSV file
def write_to_csv(filename, combined_data):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Bandwidth Rate (bps)', 'Web App Health (%)'])
        writer.writerows(combined_data)

def main():
    # Set up CLI argument parsing
    parser = argparse.ArgumentParser(description='Query Prometheus for router bandwidth and web app health, and save data to CSV.')
    
    # Define the CLI arguments
    parser.add_argument('--prometheus-url', required=True, help='The Prometheus server IP or FQDN, e.g., http://10.0.1.101:9090')
    parser.add_argument('--router-ip', required=True, help='The IP address of the router, e.g., 10.0.2.1')
    parser.add_argument('--interface-name', required=True, help='The router interface name, e.g., Fa0/0')
    parser.add_argument('--web-app-url', required=True, help='The IP address or FQDN of the web application, e.g., http://10.0.3.101')
    parser.add_argument('--days', type=int, required=True, help='The number of days to query from Prometheus')
    parser.add_argument('--step', required=True, help='The query step interval, e.g., 60s, 5m')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Get bandwidth data
    print("Querying bandwidth data...")
    bandwidth_data = get_bandwidth_data(args.prometheus_url, args.router_ip, args.interface_name, args.days, args.step)
    
    # Get web app health data
    print("Querying web app health data...")
    web_app_health_data = get_web_app_health(args.prometheus_url, args.web_app_url, args.days, args.step)
    
    # Combine the data
    print("Combining data...")
    combined_data = combine_data(bandwidth_data, web_app_health_data)
    
    # Write the data to CSV
    output_filename = f"combined_output_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    
    # Combine the folder path and the filename
    output_path = os.path.join(output_folder, output_filename)
    
    write_to_csv(output_path, combined_data)
    
    print(f"Data written to {output_filename}")

if __name__ == "__main__":
    main()