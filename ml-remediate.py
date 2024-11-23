import paramiko
import argparse
import time

def ssh_execute_command(host, username, password, commands):
    """
    Establish an SSH connection to a host and execute a list of commands.
    """
    try:
        # Establish SSH connection
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, username=username, password=password)

        # Start a shell session
        shell = ssh.invoke_shell()
        time.sleep(1)  # Allow the shell to initialize

        # Execute commands
        for command in commands:
            shell.send(command + "\n")
            time.sleep(1)  # Allow time for the command to execute
            # output = shell.recv(1024).decode()
            # print(f"Command: {command}\nOutput: {output.strip()}\n")
        
        shell.send("write memory\n")
        time.sleep(1)
        print("Configuration saved.")

        shell.close()
        ssh.close()
    except Exception as e:
        print(f"Failed to execute commands on {host}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Adjust VRRP priority on a specific router.")
    parser.add_argument('--router-ip', type=str, required=True, help="IP address of the router.")
    parser.add_argument('--username', type=str, required=True, help="SSH username for the router.")
    parser.add_argument('--password', type=str, required=True, help="SSH password for the router.")
    parser.add_argument('--interface', type=str, required=True, help="Interface name (e.g., FastEthernet0/0).")
    parser.add_argument('--vrrp-group', type=int, required=True, help="VRRP group number.")
    parser.add_argument('--vrrp-priority', type=int, required=True, help="New VRRP priority.")
    args = parser.parse_args()

    # Commands to adjust VRRP priority
    commands = [
        "enable",  # Enable privileged EXEC mode (if a password is required, additional logic is needed)
        f"configure terminal",
        f"interface {args.interface}",
        f"vrrp {args.vrrp_group} priority {args.vrrp_priority}",  # Corrected attribute reference
        "exit",
        "exit"
    ]

    print(f"Adjusting VRRP priority on router {args.router_ip}...")
    ssh_execute_command(args.router_ip, args.username, args.password, commands)
    print(f"VRRP priority adjusted on {args.router_ip}.")

if __name__ == "__main__":
    main()
