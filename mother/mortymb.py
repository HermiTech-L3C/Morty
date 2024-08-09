import os
import subprocess
import sys
import importlib.util
import requests
import zipfile
import json
import logging
from git import Repo

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def install_dependencies():
    """Ensure all required dependencies are installed."""
    dependencies = ["GitPython", "requests"]
    logging.info("Checking and installing dependencies if needed.")
    
    for dependency in dependencies:
        if importlib.util.find_spec(dependency) is None:
            logging.info(f"Installing {dependency}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dependency])
                logging.info(f"Successfully installed {dependency}.")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to install {dependency}: {e}")
                sys.exit(1)
        else:
            logging.info(f"{dependency} is already installed.")

def get_kicad_project_structure(project_name):
    """Return the standard KiCad project structure."""
    return {
        project_name: [
            f"{project_name}.kicad_pro",
            f"{project_name}.kicad_sch",
            f"{project_name}.kicad_pcb",
            "config": [
                "project_settings.json",
                "user_preferences.json"
            ],
            "schematic": [
                "system_on_chip.sch",
                "memory.sch",
                "sensors.sch",
                "wireless_communication.sch",
                "power_management.sch",
                "clocking.sch",
                "interface_connectivity.sch",
            ],
            "footprints": [
                "footprints.pretty/"
            ],
            "symbols": [
                "symbols.lib"
            ],
            "3dmodels": [],
            "output": [
                "Gerber/",
                "BOM/",
                "Step/",
                "Netlist/"
            ],
            "docs": [
                "README.md",
                "CHANGELOG.md",
                "LICENSE"
            ],
            "scripts": [
                "generate_bom.py",
                "generate_gerbers.py",
                "export_netlist.py"
            ]
        ]
    }

def create_structure(base_path, structure):
    """Recursively create the directory structure."""
    for key, value in structure.items():
        dir_path = os.path.join(base_path, key)
        try:
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Created directory: {dir_path}")
        except OSError as e:
            logging.error(f"Failed to create directory {dir_path}: {e}")
            sys.exit(1)

        for item in value:
            item_path = os.path.join(dir_path, item)
            if item.endswith('/'):
                try:
                    os.makedirs(item_path, exist_ok=True)
                    logging.info(f"Created directory: {item_path}")
                except OSError as e:
                    logging.error(f"Failed to create directory {item_path}: {e}")
                    sys.exit(1)
            else:
                try:
                    with open(item_path, 'w') as f:
                        f.write(f"This is a placeholder for {item}")
                    logging.debug(f"Created placeholder for {item}")
                except OSError as e:
                    logging.error(f"Failed to create file {item_path}: {e}")
                    sys.exit(1)

def initialize_git_repository(base_path):
    """Initialize a Git repository and make the initial commit."""
    try:
        logging.info("Initializing Git repository")
        repo = Repo.init(base_path)
        repo.index.add(repo.untracked_files)
        repo.index.commit("Initial commit with project structure")
        logging.info("Git repository initialized and initial commit made.")
    except Exception as e:
        logging.error(f"Failed to initialize Git repository: {e}")
        sys.exit(1)

def create_kicad_project_structure(project_name):
    """Create the standard KiCad project structure."""
    structure = get_kicad_project_structure(project_name)
    base_path = os.path.join(os.getcwd(), project_name)

    if os.path.exists(base_path):
        logging.error(f"Project {project_name} already exists at {base_path}.")
        sys.exit(1)

    logging.info(f"Creating project structure for {project_name} at {base_path}")
    create_structure(base_path, {project_name: structure[project_name]})
    initialize_git_repository(base_path)

def download_file(url, destination):
    """Download a file from a URL."""
    try:
        logging.info(f"Downloading from {url} to {destination}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, 'wb') as f:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            for data in response.iter_content(1024):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total_size)
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {done*2}%")
                sys.stdout.flush()
        logging.info(f"Downloaded {destination} ({total_size / (1024 * 1024):.2f} MB)")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {url}: {e}")
        sys.exit(1)

def extract_zip(zip_path, extract_to):
    """Extract a ZIP file."""
    try:
        logging.info(f"Extracting {zip_path} to {extract_to}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_path)
        logging.info(f"Extracted and removed {zip_path}")
    except zipfile.BadZipFile as e:
        logging.error(f"Failed to extract {zip_path}: {e}")
        sys.exit(1)

def download_library(library_url, destination_folder):
    """Download and extract a library from the given URL."""
    os.makedirs(destination_folder, exist_ok=True)
    zip_path = os.path.join(destination_folder, "library.zip")
    download_file(library_url, zip_path)
    extract_zip(zip_path, destination_folder)

def generate_netlist(project_name, base_path):
    """Generate the Netlist file."""
    try:
        netlist_path = os.path.join(base_path, "output", "Netlist", f"{project_name}.net")
        subprocess.run(["eeschema", "--export", netlist_path], check=True)
        logging.info(f"Netlist generated: {netlist_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to generate Netlist: {e}")
        sys.exit(1)

def generate_bom(project_name, base_path):
    """Generate the BOM file."""
    try:
        bom_path = os.path.join(base_path, "output", "BOM", f"{project_name}_BOM.xml")
        subprocess.run(["eeschema", "--export", bom_path], check=True)
        logging.info(f"BOM generated: {bom_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to generate BOM: {e}")
        sys.exit(1)

def generate_gerbers(project_name, base_path):
    """Generate the Gerber files."""
    try:
        gerber_path = os.path.join(base_path, "output", "Gerber")
        subprocess.run(["pcbnew", "--plot", f"{base_path}/{project_name}.kicad_pcb"], check=True)
        logging.info(f"Gerber files generated at: {gerber_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to generate Gerbers: {e}")
        sys.exit(1)

def generate_files(project_name):
    """Generate required files like Netlist, BOM, and Gerbers."""
    base_path = os.path.join(os.getcwd(), project_name)
    logging.info("Generating files for the project")
    generate_netlist(project_name, base_path)
    generate_bom(project_name, base_path)
    generate_gerbers(project_name, base_path)

def get_component_list():
    """Return the list of components for the BOM."""
    return [
        ["U1", "NXP i.MX 8M Mini Quad", "BGA-400", "16.00", "30.00", "0.00", "Top"],
        ["U2", "Xilinx Zynq MPSoC", "BGA-484", "48.00", "30.00", "0.00", "Top"],
        ["U3", "Micron LPDDR4 4GB", "BGA-178", "16.00", "70.00", "0.00", "Top"],
        ["U4", "Samsung eMMC 128GB", "BGA-153", "48.00", "70.00", "0.00", "Top"],
        ["U5", "Bosch BNO080", "LGA-28", "16.00", "110.00", "0.00", "Top"],
        ["U6", "Intel 9260NGW", "M.2 (2230)", "48.00", "110.00", "0.00", "Top"],
        ["U7", "Google Coral Edge TPU", "M.2 Key E", "16.00", "150.00", "0.00", "Top"],
        ["U8", "TI DRV8432", "HTSSOP-36", "48.00", "150.00", "0.00", "Top"],
        ["U9", "TI TPS65988", "VQFN-48", "16.00", "190.00", "0.00", "Top"],
        ["U10", "Analog ADP5054", "LFCSP-32", "48.00", "190.00", "0.00", "Top"],
        ["C1-C20", "100nF Capacitors", "0805", "32.00", "225.00", "0.00", "Top"],
        ["L1", "Inductor (Power Filter)", "0603", "20.00", "40.00", "0.00", "Top"],
        ["C21-C22", "22uF Capacitors", "0805", "24.00", "30.00", "0.00", "Top"],
        ["R1-R2", "10kΩ Resistors", "0603", "28.00", "40.00", "0.00", "Top"],
        ["U11", "Crystal Oscillator", "4-pin", "30.00", "20.00", "0.00", "Top"],
        ["D1", "Schottky Diode", "SOD-123", "34.00", "30.00", "0.00", "Top"],
        ["U12", "Level Shifter", "TSSOP-8", "38.00", "40.00", "0.00", "Top"],
        ["Q1", "N-Channel MOSFET", "SOT-23", "42.00", "30.00", "0.00", "Top"],
        ["C23-C24", "100uF Capacitors", "1210", "46.00", "40.00", "0.00", "Top"],
        ["R3-R4", "47Ω Resistors", "0603", "50.00", "30.00", "0.00", "Top"],
        ["U13", "RTC (Real-Time Clock)", "SOIC-8", "54.00", "40.00", "0.00", "Top"],
        ["U14", "EEPROM", "SOIC-8", "58.00", "30.00", "0.00", "Top"],
        ["J1", "Main Power Connector", "Through-hole", "62.00", "40.00", "0.00", "Top"],
        ["J2", "Programming Header", "Through-hole", "66.00", "30.00", "0.00", "Top"],
        ["J3", "UART Header", "Through-hole", "70.00", "40.00", "0.00", "Top"]
    ]

def generate_bom_csv(project_name, base_path):
    """Generate the BOM CSV file."""
    bom_path = os.path.join(base_path, "output", "BOM", f"{project_name}_BOM.csv")
    components = get_component_list()

    try:
        with open(bom_path, 'w') as f:
            f.write("Reference Designator,Component,Package,Position X (mm),Position Y (mm),Rotation (degrees),Layer\n")
            for component in components:
                f.write(",".join(component) + "\n")
        logging.info(f"BOM file generated successfully at {bom_path} with {len(components)} components.")
    except OSError as e:
        logging.error(f"Failed to generate BOM file: {e}")
        sys.exit(1)

def get_project_settings():
    """Return the default project settings."""
    return {
        "board_size": "6in x 3.5in",
        "layer_stack": ["Top Layer", "Bottom Layer", "Internal Layer 1", "Internal Layer 2"],
        "copper_thickness": "35um"
    }

def get_user_preferences():
    """Return the default user preferences."""
    return {
        "preferred_components": ["Resistor", "Capacitor", "Inductor", "Microcontroller"],
        "output_formats": ["Gerber", "STEP", "PDF"]
    }

def write_json_to_file(data, file_path):
    """Write data to a JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Successfully wrote to {file_path}")
    except OSError as e:
        logging.error(f"Failed to write to {file_path}: {e}")
        sys.exit(1)

def customize_project(project_name):
    """Customize the project settings and user preferences."""
    base_path = os.path.join(os.getcwd(), project_name, "config")
    project_settings_path = os.path.join(base_path, "project_settings.json")
    user_preferences_path = os.path.join(base_path, "user_preferences.json")

    logging.info("Customizing project settings")
    write_json_to_file(get_project_settings(), project_settings_path)

    logging.info("Customizing user preferences")
    write_json_to_file(get_user_preferences(), user_preferences_path)

def run_project_setup(project_name="MortBoard", footprints_url=None, symbols_url=None):
    """Main function to run the entire project setup."""
    logging.info("Starting project setup")
    try:
        install_dependencies()
        create_kicad_project_structure(project_name)

        if footprints_url:
            download_library(footprints_url, f"{project_name}/footprints/official_lib")
        if symbols_url:
            download_library(symbols_url, f"{project_name}/symbols/official_lib")

        customize_project(project_name)
        generate_files(project_name)
        logging.info(f"{project_name} project setup completed successfully.")
    except Exception as e:
        logging.error(f"Project setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_project_setup(
        project_name="MortBoard",
        footprints_url="https://example.com/kicad-footprints.zip",
        symbols_url="https://example.com/kicad-symbols.zip"
    )