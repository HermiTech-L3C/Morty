import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import socket
import serial
import casadi as ca

import logging as _logging

# Attempt to import PYNQ for FPGA acceleration; fall back gracefully if unavailable
try:
    from pynq import Overlay, allocate  # PYNQ library to interface with the FPGA
    _pynq_available = True
except ImportError:
    _pynq_available = False

# Load FPGA overlay only when PYNQ is present
_overlay = None
_dma = None
if _pynq_available:
    try:
        _overlay = Overlay("/path/to/your/bitstream.bit")  # Ensure the correct bitstream is loaded
        _dma = _overlay.axi_dma_0  # Access the DMA interface for data transfer
    except Exception as _e:
        _logging.warning(f"FPGA overlay unavailable: {_e}. Falling back to CPU-only mode.")
        _pynq_available = False

# Neural Network Architecture for PINN with FPGA Acceleration
class BipedalHumanoidPINN(models.Model):
    def __init__(self, use_fpga=True):
        super(BipedalHumanoidPINN, self).__init__()
        self.fc1 = layers.Dense(512, activation='relu', input_shape=(60,))
        self.fc2 = layers.Dense(512, activation='relu')
        self.fc3 = layers.Dense(256, activation='relu')
        self.fc4 = layers.Dense(60)
        self.dropout = layers.Dropout(0.4)
        self.use_fpga = use_fpga and _pynq_available

    def call(self, inputs, training=False):
        x = self.fc1(inputs)
        x = self.fc2(x)
        if training:
            x = self.dropout(x)
        x = self.fc3(x)

        if self.use_fpga:
            x = self.offload_to_fpga(x)  # Offload specific operations to FPGA

        control_signals = self.fc4(x)
        return control_signals

    def offload_to_fpga(self, data):
        # Convert TensorFlow tensor to numpy array before DMA transfer
        data_np = data.numpy() if hasattr(data, 'numpy') else np.array(data)

        # Allocate input and output buffers for DMA transfer
        input_buffer = allocate(shape=data_np.shape, dtype=np.float32)
        output_buffer = allocate(shape=data_np.shape, dtype=np.float32)

        # Copy data to input buffer
        np.copyto(input_buffer, data_np.astype(np.float32))

        # Start DMA transfer to the FPGA
        _dma.sendchannel.transfer(input_buffer)
        _dma.recvchannel.transfer(output_buffer)
        _dma.sendchannel.wait()
        _dma.recvchannel.wait()

        # Copy results from the output buffer back as a tensor
        processed_data = tf.constant(np.copy(output_buffer), dtype=tf.float32)
        return processed_data


# Dreamer World Model for high-level planning and strategic foresight
class DreamerModel(models.Model):
    """Recurrent world model that predicts high-level planning goals from current state."""

    def __init__(self, input_dim=60, action_dim=60, hidden_dim=256):
        super(DreamerModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = layers.GRU(hidden_dim, return_sequences=False)
        self.fc_goal = layers.Dense(action_dim)

    def call(self, inputs, training=False):
        # Accept either (batch, input_dim) or (batch, timesteps, input_dim)
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=1)
        h = self.gru(inputs, training=training)
        goals = self.fc_goal(h)
        return goals


# Reinforcement Learning Agent integrated with Dreamer for high-level planning
class RLAgent(models.Model):
    """Actor-critic RL agent that uses Dreamer goals to guide action selection."""

    def __init__(self, input_dim=60, action_dim=60, hidden_dim=256, dreamer_model=None,
                 goal_blend_weight=0.1):
        super(RLAgent, self).__init__()
        self.dreamer_model = dreamer_model
        # Weight controlling how strongly Dreamer goals influence the actor input.
        # A value of 0 disables Dreamer influence; 1 makes goals equal to state weight.
        self.goal_blend_weight = goal_blend_weight
        # Actor network
        self.actor_fc1 = layers.Dense(hidden_dim, activation='relu')
        self.actor_fc2 = layers.Dense(action_dim, activation='tanh')
        # Critic network
        self.critic_fc1 = layers.Dense(hidden_dim, activation='relu')
        self.critic_fc2 = layers.Dense(1)

    def actor(self, state, training=False):
        if self.dreamer_model is not None:
            goals = self.dreamer_model(state, training=training)
            # Blend Dreamer planning goals into the state representation
            combined = state + self.goal_blend_weight * goals
        else:
            combined = state
        x = self.actor_fc1(combined)
        return self.actor_fc2(x)

    def critic(self, state, training=False):
        x = self.critic_fc1(state)
        return self.critic_fc2(x)

    def call(self, inputs, training=False):
        return self.actor(inputs, training=training)


# Physics-informed loss combining data-driven and physical constraints
def physics_informed_loss(pinn_model, inputs, targets=None,
                          effort_weight=0.1, smoothness_weight=0.01, balance_weight=0.1):
    """Compute physics-informed loss with control-effort and smoothness constraints.

    Args:
        pinn_model: The PINN model instance.
        inputs: Input tensor of shape (batch, input_dim).
        targets: Optional target tensor for supervised guidance.
        effort_weight (float): Penalty weight for large control effort (default 0.1).
        smoothness_weight (float): Penalty weight for input-gradient magnitude (default 0.01).
        balance_weight (float): Penalty weight for non-zero mean control force (default 0.1).

    Returns:
        Scalar total loss tensor.
    """
    inputs_var = tf.Variable(inputs, trainable=False, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(inputs_var)
        predictions = pinn_model(inputs_var, training=True)

    grads = tape.gradient(predictions, inputs_var)

    # Data loss
    if targets is not None:
        data_loss = tf.reduce_mean(tf.square(predictions - targets))
    else:
        data_loss = tf.reduce_mean(tf.square(predictions))

    # Physics constraint: penalise large control effort
    effort_loss = tf.reduce_mean(tf.square(predictions))

    # Physics constraint: penalise rapid changes across the batch (smoothness)
    smoothness_loss = tf.reduce_mean(tf.square(grads)) if grads is not None else 0.0

    # Balance constraint: net control force should be near zero
    balance_loss = tf.square(tf.reduce_mean(predictions))

    total_loss = data_loss + effort_weight * effort_loss + smoothness_weight * smoothness_loss + balance_weight * balance_loss
    return total_loss


# Training loop coordinating PINN and RL agent with Dreamer integration
def train(pinn_model, rl_agent, optimizer_pinn, optimizer_rl, inputs, num_epochs=1000):
    """Train the PINN and RL agent jointly.

    Args:
        pinn_model: The PINN model instance.
        rl_agent: The RL agent instance.
        optimizer_pinn: Keras optimizer for the PINN.
        optimizer_rl: Keras optimizer for the RL agent.
        inputs: numpy array of shape (N, input_dim) used as training data.
        num_epochs (int): Number of training epochs.
    """
    inputs_tensor = tf.constant(inputs, dtype=tf.float32)

    for epoch in range(num_epochs):
        # --- Train PINN ---
        with tf.GradientTape() as tape_pinn:
            pinn_loss = physics_informed_loss(pinn_model, inputs_tensor)

        grads_pinn = tape_pinn.gradient(pinn_loss, pinn_model.trainable_variables)
        optimizer_pinn.apply_gradients(zip(grads_pinn, pinn_model.trainable_variables))

        # --- Train RL agent ---
        with tf.GradientTape() as tape_rl:
            actions = rl_agent(inputs_tensor, training=True)
            values = rl_agent.critic(inputs_tensor, training=True)
            # Simplified advantage actor-critic loss
            actor_loss = -tf.reduce_mean(values)
            critic_loss = tf.reduce_mean(
                tf.square(values - tf.stop_gradient(tf.reduce_mean(values)))
            )
            rl_loss = actor_loss + 0.5 * critic_loss

        grads_rl = tape_rl.gradient(rl_loss, rl_agent.trainable_variables)
        optimizer_rl.apply_gradients(zip(grads_rl, rl_agent.trainable_variables))

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}: PINN Loss = {float(pinn_loss):.4f}, RL Loss = {float(rl_loss):.4f}")


# Define the optimization problem using CasADi with integration of Dreamer planning
def define_optimization_problem(dreamer_goals):
    n_controls = 60
    u = ca.MX.sym('u', n_controls)

    # Objective: Minimize control effort while achieving goals set by Dreamer
    effort_objective = ca.mtimes(u.T, u)
    goal_tracking_objective = ca.sumsqr(u - dreamer_goals)

    objective = effort_objective + 0.1 * goal_tracking_objective  # Weighted sum of objectives

    # Define constraints (example: control signals must be within certain limits)
    lb_u = -np.ones(n_controls) * 1.0  # Lower bound for control signals
    ub_u = np.ones(n_controls) * 1.0  # Upper bound for control signals

    nlp = {'x': u, 'f': objective}
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    
    return solver, lb_u, ub_u

# Cross-communication between Dreamer and CasADi
def optimize_with_casadi(dreamer_model, inputs):
    # Get high-level goals from Dreamer
    dreamer_goals = dreamer_model(inputs)

    # Define the optimization problem based on Dreamer's goals
    solver, lb_u, ub_u = define_optimization_problem(dreamer_goals)

    # Solve the optimization problem
    sol = solver(x0=dreamer_goals, lbx=lb_u, ubx=ub_u)
    optimized_controls = sol['x'].full().flatten()

    return optimized_controls

# Initialize TCP/IP communication with ROS node
def initialize_socket_connection():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 5000))  # Replace with appropriate IP and PORT if needed
    server_socket.listen(1)
    print("Waiting for connection...")
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")
    return conn

# Initialize serial communication with FPGA
def initialize_serial(port='/dev/ttyUSB0', baudrate=9600):
    try:
        ser = serial.Serial(port, baudrate)
        return ser
    except serial.SerialException as e:
        print(f"Failed to initialize serial communication: {e}")
        return None

# Serial connection is initialised on demand (not at module import time)
_ser = None

def get_serial(port='/dev/ttyUSB0', baudrate=9600):
    """Return a cached serial connection, initialising it on first call."""
    global _ser
    if _ser is None:
        _ser = initialize_serial(port, baudrate)
    return _ser

# Send and receive data to/from FPGA
def communicate_with_fpga(control_signals, port='/dev/ttyUSB0', baudrate=9600):
    """Send control signals to FPGA over serial and return the processed response.

    Falls back to returning the original signals when serial is unavailable.
    """
    ser = get_serial(port, baudrate)
    if ser is None:
        print("Serial connection unavailable; returning original control signals.")
        return control_signals
    try:
        ser.write(control_signals.tobytes())
        optimized_control_signal = ser.read(size=240)  # 60 float32 values × 4 bytes
        optimized_control_signal = np.frombuffer(optimized_control_signal, dtype=np.float32)
        return optimized_control_signal
    except Exception as e:
        print(f"Error in communication with FPGA: {e}")
        return control_signals  # Return the original control signals if FPGA communication fails

# Main loop for receiving data from ROS node, processing it, and sending back control signals
def main():
    conn = initialize_socket_connection()
    pinn_model = BipedalHumanoidPINN()
    dreamer_model = DreamerModel(input_dim=60, action_dim=60)  # Example Dreamer model initialization
    rl_agent = RLAgent(input_dim=60, action_dim=60, dreamer_model=dreamer_model)
    optimizer_pinn = optimizers.Adam(learning_rate=0.001)
    optimizer_rl = optimizers.Adam(learning_rate=0.001)

    while True:
        try:
            data = conn.recv(240)  # Adjust size if necessary
            if not data:
                break
            inputs = np.frombuffer(data, dtype=np.float32).reshape(1, -1)

            # Use the integrated Dreamer and CasADi to optimize control signals
            optimized_control_signals = optimize_with_casadi(dreamer_model, inputs)

            # Communicate with FPGA and refine control signals
            final_control_signals = communicate_with_fpga(optimized_control_signals)

            # Send the final control signals back to the ROS node
            conn.sendall(final_control_signals.tobytes())
        except socket.error as e:
            print(f"Socket error: {e}")
            break

    conn.close()

if __name__ == "__main__":
    main()