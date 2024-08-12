import rospy
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Wrench
from std_msgs.msg import Float64MultiArray
from sklearn.preprocessing import StandardScaler
from collections import deque
import socket
import struct

# Initialize ROS node and publisher
def initialize_ros_node():
    rospy.init_node('bipedal_humanoid_pinn', anonymous=True)
    control_pub = rospy.Publisher('/control_signals', Float64MultiArray, queue_size=10)
    return control_pub

control_pub = initialize_ros_node()

# Data storage and normalization
data_storage = {
    "joint_angles": deque(maxlen=100),
    "velocities": deque(maxlen=100),
    "torques": deque(maxlen=100),
    "foot_forces": deque(maxlen=100),
    "hand_joint_angles": deque(maxlen=100),
    "object_forces": deque(maxlen=100)
}

scaler = StandardScaler()

def update_and_normalize(data, new_data):
    data.append(new_data)
    if len(data) == data.maxlen:
        normalized_data = scaler.fit_transform(np.array(data))
        return normalized_data
    return None

# ROS Callback Functions
def joint_state_callback(msg):
    try:
        joint_angles = np.array(msg.position, dtype=np.float32).reshape(1, -1)
        velocities = np.array(msg.velocity, dtype=np.float32).reshape(1, -1)
        torques = np.array(msg.effort, dtype=np.float32).reshape(1, -1)

        norm_joint_angles = update_and_normalize(data_storage["joint_angles"], joint_angles)
        norm_velocities = update_and_normalize(data_storage["velocities"], velocities)
        norm_torques = update_and_normalize(data_storage["torques"], torques)

        if norm_joint_angles is not None and norm_velocities is not None and norm_torques is not None:
            data_storage["joint_angles_normalized"] = norm_joint_angles
            data_storage["velocities_normalized"] = norm_velocities
            data_storage["torques_normalized"] = norm_torques
            send_data_to_tpu()  # Send data to TPU whenever we have a full set
    except Exception as e:
        rospy.logerr(f"Error in joint_state_callback: {e}")

def foot_force_callback(msg):
    try:
        foot_forces = np.array([msg.force.x, msg.force.y, msg.force.z], dtype=np.float32).reshape(1, -1)
        norm_foot_forces = update_and_normalize(data_storage["foot_forces"], foot_forces)
        if norm_foot_forces is not None:
            data_storage["foot_forces_normalized"] = norm_foot_forces
    except Exception as e:
        rospy.logerr(f"Error in foot_force_callback: {e}")

def hand_joint_state_callback(msg):
    try:
        hand_joint_angles = np.array(msg.position, dtype=np.float32).reshape(1, -1)
        norm_hand_joint_angles = update_and_normalize(data_storage["hand_joint_angles"], hand_joint_angles)
        if norm_hand_joint_angles is not None:
            data_storage["hand_joint_angles_normalized"] = norm_hand_joint_angles
    except Exception as e:
        rospy.logerr(f"Error in hand_joint_state_callback: {e}")

def object_force_callback(msg):
    try:
        object_forces = np.array([msg.force.x, msg.force.y, msg.force.z], dtype=np.float32).reshape(1, -1)
        norm_object_forces = update_and_normalize(data_storage["object_forces"], object_forces)
        if norm_object_forces is not None:
            data_storage["object_forces_normalized"] = norm_object_forces
    except Exception as e:
        rospy.logerr(f"Error in object_force_callback: {e}")

# Subscribe to ROS topics
def subscribe_to_ros_topics():
    rospy.Subscriber('/joint_states', JointState, joint_state_callback)
    rospy.Subscriber('/foot_force', Wrench, foot_force_callback)
    rospy.Subscriber('/hand_joint_states', JointState, hand_joint_state_callback)
    rospy.Subscriber('/object_force', Wrench, object_force_callback)

subscribe_to_ros_topics()

# Initialize TCP/IP communication with TPU
def initialize_socket_connection():
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('TPU_IP_ADDRESS', TPU_PORT))  # Replace with actual TPU IP and PORT
        return client_socket
    except socket.error as e:
        rospy.logerr(f"Failed to initialize socket connection: {e}")
        return None

client_socket = initialize_socket_connection()

# Send data to TPU and receive control signals
def send_data_to_tpu():
    if client_socket:
        try:
            input_data = np.hstack((
                data_storage["joint_angles_normalized"][-1],
                data_storage["velocities_normalized"][-1],
                data_storage["torques_normalized"][-1],
                data_storage["foot_forces_normalized"][-1],
                data_storage["hand_joint_angles_normalized"][-1],
                data_storage["object_forces_normalized"][-1]
            )).astype(np.float32)

            client_socket.sendall(input_data.tobytes())
            control_signal_data = client_socket.recv(240)  # 60 float32 values
            control_signals = np.frombuffer(control_signal_data, dtype=np.float32)
            publish_control_signals(control_signals)
        except socket.error as e:
            rospy.logerr(f"Error in communication with TPU: {e}")

# Publish control signals
def publish_control_signals(control_signals):
    control_msg = Float64MultiArray(data=control_signals)
    control_pub.publish(control_msg)

if __name__ == "__main__":
    rospy.spin()