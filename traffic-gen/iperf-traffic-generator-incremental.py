import subprocess
import time
import argparse
import sys

def run_iperf(server, bandwidth_mbps, duration):
    # Construct the iperf command with the dynamic bandwidth
    command = f"iperf3.exe -c {server} -u -b {bandwidth_mbps}M -t {duration}"
    print(f"Executing: {command}")
    # Use subprocess to execute the command
    subprocess.run(command, shell=True)

def generate_incremental_traffic(server, min_bw, max_bw, step, interval):
    try:
        current_bw = min_bw
        while current_bw <= max_bw:
            run_iperf(server, current_bw, interval)
            current_bw += step  # Increase bandwidth by step
            time.sleep(5)  # Sleep for 5 seconds between iterations
    except KeyboardInterrupt:
        print("\nTraffic generation interrupted by user (CTRL+C). Exiting...")
        sys.exit(0)

if __name__ == "__main__":
    # Argument parser to accept min_bw, max_bw, step, and interval from command-line
    parser = argparse.ArgumentParser(description="Incremental bandwidth traffic generator using iperf3")
    parser.add_argument('--server', type=str, default='127.0.0.1', help='FQDN or IP address of the iPerf server (default: 127.0.0.1)')
    parser.add_argument('--min-bw', type=float, default=1, help='Minimum bandwidth in Mbps (default: 1 Mbps)')
    parser.add_argument('--max-bw', type=float, default=30, help='Maximum bandwidth in Mbps (default: 30 Mbps)')
    parser.add_argument('--step', type=float, default=1, help='Step size for bandwidth increment in Mbps (default: 1 Mbps)')
    parser.add_argument('--interval', type=int, default=300, help='Duration for each iperf3 test in seconds (default: 300 seconds)')

    # Parse the arguments
    args = parser.parse_args()

    # Call the traffic generation function with arguments
    generate_incremental_traffic(server=args.server, min_bw=args.min_bw, max_bw=args.max_bw, step=args.step, interval=args.interval)
