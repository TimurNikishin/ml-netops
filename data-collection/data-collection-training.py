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
def date_to_unix_timestamp(date):
    return int(date.timestamp())

# Function to query Prometheus in chunks
def query_prometheus_in_chunks(prometheus_url, query, start_time, end_time, step):
    current_time = start_time
    all_data = []

    while current_time < end_time:
        next_time = min(current_time + timedelta(days=1), end_time)  # Querying in 1-day chunks
        url = generate_query_range(prometheus_url, query, date_to_unix_timestamp(current_time), date_to_unix_timestamp(next_time), step)
        response = requests.get(url)
        response_json = response.json()

        if response_json['status'] != 'success':
            print(f"Error querying Prometheus data: {response_json['error']}")
            sys.exit(1)

        if 'result' in response_json['data'] and response_json['data']['result']:
            all_data.extend(response_json['data']['result'][0]['values'])

        current_time = next_time

    return all_data

# Function to get the bandwidth data from Prometheus
def get_bandwidth_data(prometheus_url, router_ip, interface_name, start_time, end_time, step):
    query = f'rate(ifHCInOctets{{ifName="{interface_name}", instance="{router_ip}:161"}}[30s]) * 8'
    return query_prometheus_in_chunks(prometheus_url, query, start_time, end_time, step)

# Function to get the web app health from Prometheus
def get_web_app_health(prometheus_url, web_app_url, start_time, end_time, step):
    query = f'avg_over_time(probe_success{{instance="{web_app_url}"}}[30s])'
    return query_prometheus_in_chunks(prometheus_url, query, start_time, end_time, step)

# Function to combine bandwidth and web app health data
def combine_data(bandwidth_data, web_app_health_data):
    combined_data = []
    bandwidth_data_dict = {bw_timestamp: bw_value for bw_timestamp, bw_value in bandwidth_data}
    health_data_dict = {health_timestamp: health_value for health_timestamp, health_value in web_app_health_data}

    # Ensure that we include all timestamps, even if one set is missing data
    all_timestamps = sorted(set(bandwidth_data_dict.keys()).union(health_data_dict.keys()))

    for timestamp in all_timestamps:
        bw_value = bandwidth_data_dict.get(timestamp)
        health_value = health_data_dict.get(timestamp)

        # Only append if both bandwidth and health data are available for the timestamp
        if bw_value is not None and health_value is not None:
            combined_data.append([
                datetime.fromtimestamp(float(timestamp), timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
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

    # Define start and end time
    end_time = datetime.now(timezone.utc)  # Current time (now)
    start_time = end_time - timedelta(days=args.days)  # Days ago

    # Get bandwidth data
    print(f"Querying bandwidth data for the last {args.days} days...")
    bandwidth_data = get_bandwidth_data(args.prometheus_url, args.router_ip, args.interface_name, start_time, end_time, args.step)
    
    # Get web app health data
    print(f"Querying web app health data for the last {args.days} days...")
    web_app_health_data = get_web_app_health(args.prometheus_url, args.web_app_url, start_time, end_time, args.step)
    
    # Combine the data
    print("Combining data...")
    combined_data = combine_data(bandwidth_data, web_app_health_data)
    
    # Write the data to CSV
    output_filename = f"combined_output_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    
    # Combine the folder path and the filename
    output_path = os.path.join(output_folder, output_filename)
    
    write_to_csv(output_path, combined_data)
    
    print(f"Data written to {output_path}")

if __name__ == "__main__":
    main()
