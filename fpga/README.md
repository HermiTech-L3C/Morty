# FPGA Component Modules

This repository contains the Verilog files and related resources for the FPGA subfolder of the overall project. The project includes several key modules such as the CPU, FPGA core, UART communication, and top-level integration.

___
![FPGAdiagram](https://github.com/HermiTech-LLC/Morty/blob/main/Images/IMG_1620.jpeg)

For FPGA design and development, checkout [amd Vivado/ vitis Design Suites](https://www.xilinx.com/products/design-tools.html) from AMD, which offer comprehensive tools for the design, simulation, and implementation of Xilinx FPGAs.

___

## Files and Modules

### `cpu.v`

This file defines the `cpu_module`, which is responsible for the core processing tasks within the FPGA. The module includes the following main components:

- **Inputs:**
  - `clk`: Clock signal
  - `reset`: Reset signal

- **Outputs:**
  - `data_out`: Data output signal
  - `error_flag`: Error flag signal

### `fpga.v`

This file defines the `fpga_module`, which represents the main FPGA core. The module includes the following main components:

- **Inputs:**
  - `clk`: Clock signal
  - `reset`: Reset signal
  - `data_in`: Data input signal

- **Outputs:**
  - `processed_data`: Processed data output signal
  - `error_flag`: Error flag signal

### `top_L.v`

This file defines the top-level module `top_level` that integrates the CPU, FPGA core, and UART communication modules. The module includes the following main components:

- **Inputs:**
  - `clk`: Clock signal
  - `reset`: Reset signal
  - `uart_rx`: UART receive signal

- **Outputs:**
  - `uart_tx`: UART transmit signal
  - `debug_data_out`: Debug data output signal

- **Internal Signals:**
  - `cpu_data_out`: Data output from CPU
  - `fpga_data_in`: Data input to FPGA
  - `cpu_error_flag`: Error flag from CPU
  - `fpga_error_flag`: Error flag from FPGA

### `uart_comm.v`

This file defines the `uart_comm` module, which handles UART communication between the FPGA and external devices. The module includes the following main components:

- **Inputs:**
  - `clk`: Clock signal
  - `reset`: Reset signal
  - `data_in`: Data input signal
  - `uart_rx`: UART receive signal

- **Outputs:**
  - `data_out`: Data output signal
  - `uart_tx`: UART transmit signal
  - `error_flags`: Error flag signals

___

Note: For final draft production refinement assistance, consider integrating professional FPGA design services like those offered by [Fidus](https://info.fidus.com/lp-fpga-design-services?creative=696376933833&keyword=fpga%20design%20firm&matchtype=p&network=g&device=m&utm_source=696376933833&utm_medium=g&utm_term=fpga%20design%20firm&utm_content=m&utm_campaign=21181582721&utm_term=fpga%20design%20firm&utm_campaign=Search+%7C+Generic+%7C+Services&utm_source=adwords&utm_medium=ppc&hsa_acc=9000272931&hsa_cam=21181582721&hsa_grp=159600839686&hsa_ad=696376933833&hsa_src=g&hsa_tgt=kwd-414193125515&hsa_kw=fpga%20design%20firm&hsa_mt=p&hsa_net=adwords&hsa_ver=3&gad_source=1&gbraid=0AAAAACxcqE3NiUNvQ3gzuArYRaK4tG-25&gclid=CjwKCAjwnei0BhB-EiwAA2xuBvVj1Cp6dZias-pLGoSc0NX0SperLy1BU1TKF-iuC6H5jWIrXzkdSRoCHJIQAvD_BwE). This service is ideal for companies needing expertise in FPGA development to ensure high-quality, reliable solutions for complex design needs.