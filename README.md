# Bipedal Humanoid Control System

___
![img](https://github.com/HermiTech-LLC/Morty/blob/main/Images/Mort.jpg)
___
# NOTE: *DOCUMENTATION NEEDS UPDATING.*

## Table of Contents
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Setup](#setup)
  - [Install Dependencies](#install-dependencies)
  - [Compile the Verilog Modules](#compile-the-verilog-modules)
  - [Run the ROS Node](#run-the-ros-node)
  - [Automate the Process](#automate-the-process)
- [Usage](#usage)
- [Detailed Descriptions](#detailed-descriptions)
  - [`rosnode.py`](#rosnodepy)
  - [`tpu.py`](#tpupy)
  - [`uart_comm.v`](#uart_commv)
  - [`top_L.v`](#top_lv)
  - [`cpu.v`](#cpuv)
  - [`fpga.v`](#fpgav)
  - [`main.py`](#mainpy)
  - [`mortymb.py`](#mortymbpy)
- [Appendix](#appendix)
  - [Power and Ground Connections](#power-and-ground-connections)
  - [Helper Function](#helper-function)
  - [Components](#components)
  - [Connections](#connections)
  - [Decoupling Capacitors](#decoupling-capacitors)
  - [GPIO Integration Plans](#gpio-integration-plans)
- [Conclusion](#conclusion)

## Overview

This project implements a bipedal humanoid control system using a Physics-Informed Neural Network (PINN) and Reinforcement Learning (RL) for stability and manipulation tasks. The system integrates with ROS for real-time control and utilizes FPGA for hardware acceleration.

## Directory Structure

- **`rosnode.py`**: Located in `/mother/Software_Firmware/rosnode.py`. This ROS node script is responsible for subscribing to sensor data topics, processing data, and publishing control signals.
- **`tpu.py`**: Located in the root directory, this script handles the Physics-Informed Neural Network (PINN) and Reinforcement Learning (RL) logic, communication with FPGA, and optimization.
- **`uart_comm.v`**: Verilog module for UART communication with FPGA.
- **`top_L.v`**: Top-level Verilog module integrating all components.
- **`cpu.v`**: Verilog module for the CPU.
- **`fpga.v`**: Verilog module for the FPGA.
- **`main.py`**: Script to compile the Verilog modules and run the ROS node.
- **`mortymb.py`**: Defines the electronic components and connections for the robotâs motherboard using `skidl`.
- **`README.md`**: This file.

## Dependencies

- ROS (Robot Operating System)
- Python 3.x
- TensorFlow
- CasADi
- scikit-learn
- iverilog (for Verilog compilation)
- KiCad (for electronic design)

## Setup

### Install Dependencies

Ensure all dependencies are installed using the following commands:
```sh
sudo apt-get install ros-humble-desktop-full
sudo apt-get install iverilog kicad
pip install tensorflow casadi scikit-learn
```

### Compile the Verilog Modules

Compile the Verilog modules using `iverilog` and `vvp`:
```sh
iverilog -o uart_comm uart_comm.v
iverilog -o top_L top_L.v
iverilog -o cpu cpu.v
iverilog -o fpga fpga.v
vvp uart_comm
vvp top_L
vvp cpu
vvp fpga
```

### Run the ROS Node

Start the ROS node that runs the system:
```sh
rosrun bipedal_control rosnode.py
```

### Automate the Process

Alternatively, use the `main.py` script to compile the Verilog modules and run the ROS node:
```sh
python main.py
```

## Usage

The system is designed to control a bipedal humanoid robot. It subscribes to ROS topics for sensor data, including joint states and forces, and publishes control signals to actuate the robot's movements. The PINN model ensures stability and manipulation capabilities, while the RL agent optimizes control strategies over time.

## Detailed Descriptions

### `rosnode.py`

- **Location**: `/mother/Software_Firmware/rosnode.py`
- **Purpose**: Acts as the interface between ROS and the TPU module. It handles the collection, normalization, and transmission of sensor data to the TPU.
- **Functionality**:
  - Initializes the ROS node and subscribers for sensor data.
  - Normalizes sensor data using `StandardScaler`.
  - Sends normalized data to the TPU module over a TCP/IP connection.
  - Publishes the received control signals back to ROS for robot actuation.

### `tpu.py`

- **Location**: Root directory
- **Purpose**: Implements the main control algorithms using the Physics-Informed Neural Network (PINN) and Reinforcement Learning (RL). It also handles communication with the FPGA.
- **Functionality**:
  - **PINN Architecture**: A deep neural network designed for stability and manipulation tasks.
  - **Reinforcement Learning**: An RL agent to optimize control strategies over time.
  - **Optimization**: Uses CasADi to define and solve optimization problems for control signals.
  - **Communication**: Establishes serial communication with the FPGA and TCP/IP communication with the ROS node.

### `uart_comm.v`

- **Purpose**: Defines the Verilog module for UART communication with the FPGA.
- **Functionality**:
  - Implements UART communication protocols.
  - Facilitates data exchange between the FPGA and other components.

### `top_L.v`

- **Purpose**: Defines the top-level Verilog module integrating the CPU, FPGA, and UART modules.
- **Functionality**:
  - Connects the CPU, FPGA, and UART communication.
  - Manages data flow between components.

### `cpu.v`

- **Purpose**: Defines the Verilog module for the CPU.
- **Functionality**:
  - Implements the CPU operations and interfaces with the UART and FPGA modules.

### `fpga.v`

- **Purpose**: Defines the Verilog module for the FPGA.
- **Functionality**:
  - Implements FPGA operations and interfaces with the CPU and UART modules.

### `main.py`

- **Purpose**: Automates the compilation of Verilog modules, execution of the ROS node, starts systemd services, and runs MicroPython scripts for centralized control interfacing.
- **Functionality**:
  - Compiles the Verilog modules using `iverilog` and `vvp`.
  - Runs the `rosnode.py` script using `rosrun`.
  - Starts specified systemd services using `systemctl`.
  - Executes a specified MicroPython script for centralized control interfacing.

## Schematics and Assembly

### `mortymb.py`

- **Purpose**: Defines and connects the electronic components of the robotâs motherboard using `skidl` for electronic design automation.
- **Functionality**:
  - **Project Structure**: Creates a standard KiCad project structure with directories for schematic files, footprints, symbols, 3D models, and output files (Gerber, BOM, Netlist).
  - **Component Definition and Connection**: Defines components such as the NXP i.MX 8M Mini Quad, Xilinx Zynq MPSoC, and others, and establishes power and data connections between them.
  - **File Generation**: Generates netlist, BOM, Gerber files, and exports these in the appropriate formats.
  - **Library Integration**: Downloads and integrates component libraries (footprints and symbols) from provided URLs.
  - **Git Integration**: Initializes a Git repository and commits the initial project structure.
  - **Customization**: Configures project settings and user preferences for the KiCad project.

## Appendix

*will be revised*

### Power and Ground Connections

- **GND**: Ground.
- **i.MX8M_VCC**: Power supply for the NXP i.MX 8M Mini Quad.
- **ZYNQ_VCC**: Power supply for the Xilinx Zynq MPSoC.
- **LPDDR4_VCC**: Power supply for the Micron LPDDR4 4GB.
- **EMMC_VCC**: Power supply for the Samsung eMMC 128GB.
- **IMU_VCC**: Power supply for the Bosch BNO080.
- **WIFI_BT_VCC**: Power supply for the Intel 9260NGW.
- **TPU_VCC**: Power supply for the Google Coral Edge TPU.
- **DRV_VCC**: Power supply for the TI DRV8432.
- **USB_VCC**: Power supply for USB ports.
- **ETH_VCC**: Power supply for the Ethernet controller and port.
- **UART_VCC**: Power supply for UART communication module.
- **PMIC_VCC**: Power supply from the Power Management IC.
- **CLOCK_VCC**: Power supply for the Crystal Oscillator.
- **FLASH_VCC**: Power supply for Flash memory.

### Helper Function

- **add_decoupling_caps(part, pin_name, gnd, num_caps=2)**:
  - Adds decoupling capacitors to a specified part and pin.
  - Decoupling capacitors help stabilize the power supply to the components.

### Components

The script defines the following components:

- **U1**: NXP i.MX 8M Mini Quad, BGA-400.
- **U2**: Xilinx Zynq MPSoC, BGA-484.
- **U3**: Micron LPDDR4 4GB, BGA-178.
- **U4**: Samsung eMMC 128GB, BGA-153.
- **U5**: Bosch BNO080, LGA-28.
- **U6**: Intel 9260NGW, M.2 (2230).
- **U7**: Google Coral Edge TPU, M.2 Key E.
- **U8**: TI DRV8432, HTSSOP-36.
- **U9**: TI TPS65988, VQFN-48.
- **U10**: Analog ADP5054, LFCSP-32.
- **C1-C20**: 100nF Capacitors, 0805.
- **L1**: Inductor (Power Filter), 0603.
- **C21-C22**: 22uF Capacitors, 0805.
- **R1-R2**: 10kΩ Resistors, 0603.
- **U11**: Crystal Oscillator, 4-pin.
- **D1**: Schottky Diode, SOD-123.
- **U12**: Level Shifter, TSSOP-8.
- **Q1**: N-Channel MOSFET, SOT-23.
- **C23-C24**: 100uF Capacitors, 1210.
- **R3-R4**: 47Ω Resistors, 0603.
- **U13**: RTC (Real-Time Clock), SOIC-8.
- **U14**: EEPROM, SOIC-8.
- **J1**: Main Power Connector, Through-hole.
- **J2**: Programming Header, Through-hole.
- **J3**: UART Header, Through-hole.

### Connections

The `mortymb.py` script establishes connections for power and data lines between components:

- **Power and Ground Connections**:
  - Connects each component’s power and ground pins to the respective power nets.
  - Example: `vcc += cpu['VCC']`.

- **CPU and RAM**:
  - Connects address and data lines between the CPU and RAM modules.
  - Example: `cpu['AD0', 'AD1', ...] += ram['DQ0', 'DQ1', ...]`.

- **FPGA**:
  - Connects FPGA power and ground pins.
  - Example: `vcc += fpga['VCC']`.

- **UART Communication**:
  - Connects UART communication lines between the ESP32, CPU, and UART module.
  - Example: `esp32['GPIO1'] += uart['T1IN']`.

- **ESP32 to USB**:
  - Connects data lines between ESP32 and USB controller.
  - Example: `esp32['GPIO21'] += usb['DP']`.

- **ESP32 to Ethernet**:
  - Connects data lines between ESP32 and Ethernet controller.
  - Example: `esp32['GPIO0'] += eth['TXEN']`.

- **PMIC**:
  - Connects the PMIC to provide regulated power to various components.
  - Example: `pmic['OUT1'] += cpu['VCC']`.

### GPIO Capabilities

The current design includes GPIO (General Purpose Input/Output) integration for enhanced sensor interfacing and integral control capabilities, making the system versatile for future expansions and customizations.

## Conclusion

This project aims to provide a comprehensive system for controlling a bipedal humanoid robot. By integrating advanced hardware with sophisticated software algorithms, the robot can achieve stable and optimized performance in real-time environments.