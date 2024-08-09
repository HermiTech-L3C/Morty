# Morty Project: Advanced Robotics Motherboard Design

## Overview

The Morty Project is dedicated to developing an advanced, small-form-factor robotics motherboard, optimized for high performance within constrained dimensions. The motherboard is designed to fit inside a 7 1/2 x 2 1/2 inch enclosure and will serve as the central hub for a sophisticated robot incorporating ROS (Robot Operating System) for real-time control, advanced AI processing, and seamless connectivity.

This README provides a comprehensive guide to the project's design, focusing on high-efficiency power management, cutting-edge components, and modularity, making it suitable for complex robotic applications.

## Key Features

- **High Performance**: Integration of the NXP i.MX 8M Mini Quad SoC and Xilinx Zynq UltraScale+ MPSoC for robust processing and FPGA-based real-time control.
- **Efficient Power Management**: Utilizes Texas Instruments PMIC and high-efficiency voltage regulators to ensure stable operation under varied loads.
- **Compact and Modular Design**: Optimized to fit within a 7 1/2 x 2 1/2 inch tin, using a 6-layer PCB for optimal space utilization, thermal efficiency, and signal integrity.
- **AI and Connectivity**: Incorporates Google Coral Edge TPU for AI tasks and Intel 9260NGW for robust Wi-Fi and Bluetooth connectivity.

## Conceptual Representation

![mortDiagram](https://github.com/HermiTech-LLC/Morty/blob/main/Images/MortBoard%20-%20for%20Paris%20Cookie%20Tin.png)

## Power Distribution

### Main Power Input

- **Primary Voltage:** 12V
- **Source:** External Power Supply

### Power Regulation

The main 12V input is regulated down to the required voltages for various components using advanced Voltage Regulator Modules (VRMs) and a Power Management IC (PMIC). The power distribution is carefully designed to ensure efficient operation of high-performance components while minimizing heat generation and energy loss.

### Components and Voltage Requirements

1. **NXP i.MX 8M Mini Quad SoC**
   - **Voltage:** 3.3V (I/O), 5V (peripherals)
   - **Current:** 0.7A
   - **Power Consumption:** 2.31W

2. **Xilinx Zynq UltraScale+ MPSoC**
   - **Core Voltage:** 0.85V / 1.2V
   - **Core Current:** 1.2A
   - **I/O Voltage:** 3.3V
   - **I/O Current:** 0.5A
   - **Power Consumption:** 2.67W (1.02W core + 1.65W I/O)

3. **Micron LPDDR4 4GB**
   - **Voltage:** 1.1V
   - **Current:** 0.4A
   - **Power Consumption:** 0.44W

4. **Samsung eMMC 128GB**
   - **Voltage:** 3.3V
   - **Current:** 0.2A
   - **Power Consumption:** 0.66W

5. **Bosch BNO080 (9-axis IMU)**
   - **Voltage:** 3.3V
   - **Current:** 0.03A
   - **Power Consumption:** 0.1W

6. **Intel 9260NGW (Wi-Fi/Bluetooth Module)**
   - **Voltage:** 3.3V
   - **Current:** 0.4A
   - **Power Consumption:** 1.32W

7. **Google Coral Edge TPU**
   - **Voltage:** 5V
   - **Current:** 0.8A
   - **Power Consumption:** 4W

8. **Motor Drivers (Texas Instruments DRV8432)**
   - **Input Voltage:** 24V (down-regulated for motors)
   - **Power Consumption:** 3.3W per motor (varies with load)

9. **Power Management IC (Texas Instruments TPS65988)**
   - **Input Voltage:** 12V
   - **Output Voltages:** 3.3V, 5V, 1.2V, 0.85V
   - **Power Consumption:** Estimated at 10W for overall system regulation

10. **Analog Devices ADP5054 (Quad Output DC-DC Converter)**
    - **Output Voltages:** 5V, 3.3V, 1.1V
    - **Power Consumption:** Estimated at 3W

### Total Power Consumption

- **3.3V Rail:** 6.039W (for SoC, FPGA I/O, eMMC, IMU, Wi-Fi/Bluetooth)
- **1.2V Rail:** 1.44W (for FPGA core)
- **0.85V Rail:** 1.02W (for FPGA core)
- **5V Rail:** 4W (for Coral Edge TPU and SoC peripherals)
- **24V Rail (Motors):** 6.6W (assuming two motors under load)
- **Overall System Power:** Approximately 32W (including overhead and PMIC)

### PCB Design and Layout

- **Size**: The PCB is designed to fit within the 7 1/2 x 2 1/2 inch tin, utilizing a 6-layer board to optimize space while ensuring thermal and electrical performance.
- **Thermal Management**: Copper pours, thermal vias, and heat sinks are integrated to manage heat from high-power components like the SoC and FPGA.
- **Modularity**: Key components are placed on separate, interconnected PCBs to allow for easier upgrades, maintenance, and scalability.

## Assembly and Integration Plan

1. **Prototype Development**: Begin with a functional prototype using larger development boards to validate the design and software integration, especially within the ROS environment.
2. **Miniaturization**: Transition to the final compact PCB design, ensuring all components fit within the space constraints of the 7 1/2 x 2 1/2 inch tin.
3. **Testing**: Conduct extensive testing, including thermal, power, and performance assessments, to ensure the board meets all design criteria.
4. **Final Assembly**: Secure the PCB within the tin using 3D-printed mounts, connect all peripherals, and perform final system calibration to ensure optimal performance.
