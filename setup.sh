#!/bin/bash

# ObserveGuard Setup Script
# Automated setup for Host Machine (Ubuntu 22.04 / WSL2)
# Usage: bash setup.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ObserveGuard Project Setup${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Detect OS
OS_TYPE="$(uname -s)"
case "$OS_TYPE" in
    Linux*)     OS="Linux";;
    Darwin*)    OS="Mac";;
    CYGWIN*)    OS="Cygwin";;
    MINGW*)     OS="MinGw";;
    *)          OS="UNKNOWN";;
esac

echo -e "${YELLOW}Detected OS: $OS${NC}\n"

# Check Python version
echo -e "${YELLOW}Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}Python $PYTHON_VERSION found${NC}"
else
    echo -e "${RED}Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

# Check Python version is 3.10+
PYTHON_VERSION_MAJOR=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1)
PYTHON_VERSION_MINOR=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f2)

if [ "$PYTHON_VERSION_MAJOR" -lt 3 ] || [ "$PYTHON_VERSION_MINOR" -lt 10 ]; then
    echo -e "${RED}Python 3.10+ required (found $PYTHON_VERSION_MAJOR.$PYTHON_VERSION_MINOR)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version compatible${NC}\n"

# Setup Conda environment (if conda available)
echo -e "${YELLOW}Setting up Python environment...${NC}"

if command -v conda &> /dev/null; then
    echo -e "${GREEN}Conda found. Creating conda environment...${NC}"
    ENV_NAME="observeguard"
    
    # Create or update conda environment
    if conda env list | grep -q "^$ENV_NAME "; then
        echo -e "${YELLOW}Environment '$ENV_NAME' exists. Updating...${NC}"
        conda activate $ENV_NAME
    else
        echo -e "${GREEN}Creating new conda environment: $ENV_NAME${NC}"
        conda create -n $ENV_NAME python=3.10 -y
        conda activate $ENV_NAME
    fi
else
    echo -e "${YELLOW}Conda not found. Using venv...${NC}"
    
    if [ ! -d "venv" ]; then
        echo -e "${GREEN}Creating virtual environment...${NC}"
        python3 -m venv venv
    fi
    
    # Activate venv
    if [ "$OS" == "Windows" ] || [ "$OS" == "Cygwin" ] || [ "$OS" == "MinGw" ]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
fi

echo -e "${GREEN}✓ Python environment ready${NC}\n"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ pip upgraded${NC}\n"

# Install PyTorch (CPU or GPU)
echo -e "${YELLOW}Installing PyTorch...${NC}"

if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected. Installing GPU-enabled PyTorch...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo -e "${YELLOW}No NVIDIA GPU detected. Installing CPU-only PyTorch...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo -e "${GREEN}✓ PyTorch installed${NC}\n"

# Install requirements
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}\n"

# Create directories
echo -e "${YELLOW}Creating project directories...${NC}"
mkdir -p data/{osworld,ssv2,probes}
mkdir -p results logs models/.cache
echo -e "${GREEN}✓ Directories created${NC}\n"

# Download datasets (optional)
echo -e "${YELLOW}Prepare datasets? (y/n)${NC}"
read -r -n 1 RESPONSE
echo

if [[ $RESPONSE =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Downloading OSWorld tasks...${NC}"
    python3 -c "
from datasets import download_osworld
result = download_osworld(output_dir='./data/osworld', prepare_splits=True, augment_tasks=True)
print('[✓] OSWorld ready')
"
    
    echo -e "${YELLOW}Preparing SSv2 dataset...${NC}"
    python3 -c "
from datasets import prepare_ssv2_dataset
result = prepare_ssv2_dataset(output_dir='./data/ssv2')
print('[✓] SSv2 ready')
"
    
    echo -e "${YELLOW}Generating probes...${NC}"
    python3 -c "
from datasets import generate_probe_suite
suite = generate_probe_suite(output_dir='./data/probes', probes_per_type=5)
print('[✓] Probes generated')
"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo -e "${YELLOW}Next steps:${NC}"
echo "1. Activate environment:"
if command -v conda &> /dev/null; then
    echo "   conda activate observeguard"
else
    echo "   source venv/bin/activate"
fi
echo ""
echo "2. Run experiments:"
echo "   python evaluation/run_osworld.py --agent observe_guard --track-energy"
echo "   python evaluation/run_ssv2_drift.py --agent observe_guard"
echo "   python evaluation/attack_simulator.py --mode compare"
echo ""
echo "3. View results in ./results/"
echo ""

# Create activation script
if [ "$OS" == "Windows" ] || [ "$OS" == "Cygwin" ] || [ "$OS" == "MinGw" ]; then
    ACTIVATE_CMD="venv\\Scripts\\activate"
else
    ACTIVATE_CMD="source venv/bin/activate"
fi

echo -e "${GREEN}✓ ObserveGuard is ready!${NC}"
