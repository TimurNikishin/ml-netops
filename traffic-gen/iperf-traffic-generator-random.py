import subprocess
import random
import time
import argparse
import sys

def run_iperf(server, bandwidth_mbps, duration):
    # Construct the iperf command with the dynamic bandwidth
    command = f"iperf3.exe -c {server} -u -b {bandwidth_mbps}M -t {duration}"
    print(f"Executing: {command}")
    # Use subprocess to execute the command
    subprocess.run(command, shell=True)

def generate_random_traffic(server, min_bw, max_bw, interval):
    try:
        while True:
            # Generate a random floating-point bandwidth between min_bw and max_bw
            random_bw = round(random.uniform(min_bw, max_bw), 2)  # Two decimal places for precision
            run_iperf(server, random_bw, interval)
            # Sleep for 5 seconds between iterations
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nTraffic generation interrupted by user (CTRL+C). Exiting...")
        sys.exit(0)

if __name__ == "__main__":
    # Argument parser to accept min_bw, max_bw, and interval from command-line
    parser = argparse.ArgumentParser(description="Random bandwidth traffic generator using iperf3")
    parser.add_argument('--server', type=str, default='127.0.0.1', help='FQDN or IP address of the iPerf server (default: 127.0.0.1)')
    parser.add_argument('--min-bw', type=float, default=1, help='Minimum bandwidth in Mbps (default: 1 Mbps)')
    parser.add_argument('--max-bw', type=float, default=30, help='Maximum bandwidth in Mbps (default: 30 Mbps)')
    parser.add_argument('--interval', type=int, default=300, help='Duration for each iperf3 test in seconds (default: 300 seconds)')

    # Parse the arguments
    args = parser.parse_args()

    # Call the traffic generation function with arguments
    generate_random_traffic(server=args.server, min_bw=args.min_bw, max_bw=args.max_bw, interval=args.interval)
