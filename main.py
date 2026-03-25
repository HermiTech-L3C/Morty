import os
import subprocess
import numpy as np
import rospy
from tensorflow.keras import optimizers
from tpu import BipedalHumanoidPINN, DreamerModel, RLAgent, physics_informed_loss, train, communicate_with_fpga
from mother.Software_Firmware.rosnode import initialize_ros_node, subscribe_to_ros_topics

def compile_verilog():
    """
    Compiles the Verilog module using iverilog and executes the compiled output using vvp.
    
    This function runs the iverilog command to compile the uart_comm.v Verilog file and then
    runs the compiled output using the vvp command. It prints a success message if the
    compilation and execution are successful. Otherwise, it prints an error message.
    
    Raises:
        subprocess.CalledProcessError: If there is an error during the Verilog compilation
                                       or execution process.
    """
    try:
        subprocess.run(['iverilog', '-o', 'uart_comm', 'uart_comm.v'], check=True)
        subprocess.run(['vvp', 'uart_comm'], check=True)
        print("Verilog module compiled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during Verilog compilation: {e}")

def run_ros_pinn():
    """
    Runs the ROS PINN node using rosrun.
    
    This function runs the rospinn.py script using the rosrun command. It prints a success
    message if the ROS node execution is successful. Otherwise, it prints an error message.
    
    Raises:
        subprocess.CalledProcessError: If there is an error during the ROS PINN execution process.
    """
    try:
        subprocess.run(['rosrun', 'rospinn', 'rospinn.py'], check=True, cwd='mother/Software_Firmware')
        print("ROS PINN node executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during ROS PINN execution: {e}")

def start_systemd_service(service_name):
    """
    Starts a specified systemd service using systemctl.
    
    Args:
        service_name (str): The name of the systemd service to start.
    
    This function runs the systemctl start command to start the specified systemd service.
    It prints a success message if the service is started successfully. Otherwise, it
    prints an error message.
    
    Raises:
        subprocess.CalledProcessError: If there is an error starting the systemd service.
    """
    try:
        subprocess.run(['systemctl', 'start', service_name], check=True)
        print(f"Systemd service {service_name} started successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error starting systemd service {service_name}: {e}")

def run_micropython(script_path):
    """
    Executes a specified MicroPython script using the micropython command.
    
    Args:
        script_path (str): The path to the MicroPython script to execute.
    
    This function runs the specified MicroPython script using the micropython command.
    It prints a success message if the script is executed successfully. Otherwise, it
    prints an error message.
    
    Raises:
        subprocess.CalledProcessError: If there is an error executing the MicroPython script.
    """
    try:
        subprocess.run(['micropython', script_path], check=True)
        print(f"MicroPython script {script_path} executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing MicroPython script {script_path}: {e}")

def setup_environment():
    """
    Performs the initial setup, including Verilog compilation, systemd service startup,
    and initializing ROS nodes and topics. Also sets up the environment for the PINN model and FPGA.
    """
    # Compile Verilog
    compile_verilog()

    # Start systemd services
    systemd_services = ['service1', 'service2']  # Replace with actual service names
    for service in systemd_services:
        start_systemd_service(service)
    
    # Run ROS PINN script
    run_ros_pinn()

    # Run MicroPython script
    micropython_script_path = 'control_interface.py'  # Replace with actual script path
    run_micropython(micropython_script_path)

    # Initialize ROS node and subscribe to topics
    control_pub = initialize_ros_node()
    subscribe_to_ros_topics()

def initialize_models_and_optimizers():
    """
    Initializes the PINN model, RL agent, and their respective optimizers.
    
    Returns:
        tuple: Initialized models and optimizers (pinn_model, rl_agent, optimizer_pinn, optimizer_rl).
    """
    pinn_model = BipedalHumanoidPINN(use_fpga=True)
    dreamer_model = DreamerModel(input_dim=60, action_dim=60)  # Example Dreamer model initialization
    rl_agent = RLAgent(input_dim=60, action_dim=60, dreamer_model=dreamer_model)
    optimizer_pinn = optimizers.Adam(learning_rate=0.001)
    optimizer_rl = optimizers.Adam(learning_rate=0.001)
    
    return pinn_model, rl_agent, optimizer_pinn, optimizer_rl

def orchestrate_dreamer_training(pinn_model, rl_agent, optimizer_pinn, optimizer_rl, inputs, num_epochs=1000):
    """
    Trains the PINN and RL agent with the integration of Dreamer model, ensuring that high-level
    planning from Dreamer is incorporated into the training process.

    Args:
        pinn_model: The PINN model instance.
        rl_agent: The RL agent instance.
        optimizer_pinn: Optimizer for the PINN model.
        optimizer_rl: Optimizer for the RL agent.
        inputs: Input data for training.
        num_epochs (int): Number of epochs for training.
    """
    train(pinn_model, rl_agent, optimizer_pinn, optimizer_rl, inputs, num_epochs)

def main():
    setup_environment()

    # Initialize models and optimizers
    pinn_model, rl_agent, optimizer_pinn, optimizer_rl = initialize_models_and_optimizers()

    # Replace 'inputs' with actual input data
    inputs = np.random.rand(100, 60).astype(np.float32)  # Example input data

    # Orchestrate the training process with Dreamer integration
    orchestrate_dreamer_training(pinn_model, rl_agent, optimizer_pinn, optimizer_rl, inputs, num_epochs=1000)

    # Example usage of communicate_with_fpga (serial init handled internally)
    sensor_data = np.random.rand(60).astype(np.float32)  # Example sensor data
    control_signal = communicate_with_fpga(sensor_data)

    # Run ROS node main loop
    rospy.spin()

if __name__ == "__main__":
    main()
