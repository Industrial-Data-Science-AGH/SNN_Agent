# Wake-Up AI: Hybrid Neuromorphic Edge System

This project implements a hybrid architecture for ultra-low power "Always-On" Edge AI, combining analog neuromorphic circuits (Layer 1) with digital software agents (Layer 3).

## Directory Structure

### 1. `hardware/` (Layer 1 & 2: The Physical World)
Focuses on the custom analog neuron implementation and PCB design.
- **`analog_core/`**: Custom analog circuit designs.
  - `schematics/`: Circuit schematics.
  - `spice_simulations/`: SPICE simulations of sub-threshold MOSFET neurons.
- **`pcb/`**: Printed Circuit Board designs (KiCad/Altium).
- **`testing/`**: Measurements and characterization logs from physical prototypes.

### 2. `neuromorphic_core/` (Spiking Neural Networks)
Tools for training and simulating the SNN models that run on the analog hardware.
- **`training/`**: Python scripts using frameworks like **Rockpool**, **Sinabs**, or **Nengo**.
- **`models/`**: Trained SNN weights and model definitions.

### 3. `firmware/` (Layer 2: The Bridge)
Code running on the microcontroller or FPGA that interfaces between the analog core and the high-level software.
- **`mcu_bridge/`**: C/C++ firmware (e.g., STM32, ESP32) reading spikes via ADC/GPIO and implementing the "Wake-Up" logic.
- **`fpga_logic/`**: Verilog/VHDL code if FPGA emulation is used.

### 4. `software/` (Layer 3: The Reactor)
High-level digital processing that runs on the host (Jetson Orin/RPi 5) when woken up.
- **`host_agent/`**: The "Reactor" - LLM/VLM (Llama 3, etc.) logic.
  - `rag/`: Retrieval-Augmented Generation module.
- **`drivers/`**: Python/C++ drivers for host-to-hardware communication (SPI/I2C/GPIO).
- **`web_dashboard/`**: Visualization tools.
  - `ui/`: Frontend (React).
  - `api/`: Backend API (FastAPI).
- **`infra/`**: Docker and deployment configuration.

### 5. `docs/`
Project documentation and references.
- Contains the architectural proposal: `Projekt Neuromorficzny Wake-Up AI.pdf`
