#!/bin/bash

# Jigsaw Competition - Reddit Comment Rule Violation Prediction
# Main execution script

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_message() {
    echo -e "${2}${1}${NC}"
}

# Check if Python is installed
if ! command -v python &> /dev/null; then
    print_message "Error: Python is not installed. Please install Python 3.8 or higher." $RED
    exit 1
fi

# Check Python version
python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_message "Error: Python $required_version or higher is required. Found: $python_version" $RED
    exit 1
fi

# Function to display help
show_help() {
    echo "Usage: ./run.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup        Install dependencies and prepare environment"
    echo "  train        Run training pipeline"
    echo "  inference    Run inference on test data"
    echo "  experiment   Run full experiment pipeline"
    echo "  test         Run unit tests"
    echo "  augment      Run data augmentation only"
    echo "  ensemble     Run ensemble inference"
    echo "  help         Show this help message"
    echo ""
    echo "Options:"
    echo "  --config     Specify configuration file (default: baseline_v2)"
    echo "  --gpu        GPU device to use (default: 0)"
    echo "  --output     Output directory (default: ./outputs)"
    echo ""
    echo "Examples:"
    echo "  ./run.sh setup"
    echo "  ./run.sh train --config baseline_v2"
    echo "  ./run.sh experiment"
    echo "  ./run.sh test"
}

# Function to setup environment
setup_environment() {
    print_message "Setting up environment..." $YELLOW
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_message "Creating virtual environment..." $GREEN
        python -m venv venv
    fi
    
    # Activate virtual environment
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    # Upgrade pip
    print_message "Upgrading pip..." $GREEN
    pip install --upgrade pip
    
    # Install requirements
    print_message "Installing requirements..." $GREEN
    pip install -r requirements.txt
    
    # Create necessary directories
    mkdir -p outputs
    mkdir -p models
    mkdir -p experiments
    mkdir -p data
    
    print_message "Setup complete!" $GREEN
}

# Function to run training
run_training() {
    print_message "Starting training..." $YELLOW
    
    config=${2:-baseline_v2}
    gpu=${4:-0}
    
    export CUDA_VISIBLE_DEVICES=$gpu
    
    # Check if data exists
    if [ ! -f "train.csv" ]; then
        print_message "Error: train.csv not found. Please place training data in current directory." $RED
        exit 1
    fi
    
    # Run training
    cd src
    python improved_training.py --config $config
    cd ..
    
    print_message "Training complete!" $GREEN
}

# Function to run inference
run_inference() {
    print_message "Starting inference..." $YELLOW
    
    gpu=${2:-0}
    export CUDA_VISIBLE_DEVICES=$gpu
    
    # Check if test data exists
    if [ ! -f "test.csv" ]; then
        print_message "Error: test.csv not found. Please place test data in current directory." $RED
        exit 1
    fi
    
    # Run inference
    cd src
    python inference.py
    cd ..
    
    print_message "Inference complete! Check submission.csv" $GREEN
}

# Function to run experiments
run_experiments() {
    print_message "Starting full experiment pipeline..." $YELLOW
    
    cd src
    python run_experiments.py
    cd ..
    
    print_message "Experiments complete!" $GREEN
}

# Function to run tests
run_tests() {
    print_message "Running unit tests..." $YELLOW
    
    # Run pytest
    python -m pytest test/ -v --tb=short
    
    print_message "Tests complete!" $GREEN
}

# Function to run data augmentation
run_augmentation() {
    print_message "Running data augmentation..." $YELLOW
    
    cd src
    python data_augmentation.py
    cd ..
    
    print_message "Data augmentation complete!" $GREEN
}

# Function to run ensemble
run_ensemble() {
    print_message "Running ensemble inference..." $YELLOW
    
    cd src
    python ensemble_inference.py --auto_ensemble --top_k 3 --use_tta
    cd ..
    
    print_message "Ensemble inference complete!" $GREEN
}

# Main script logic
case "$1" in
    setup)
        setup_environment
        ;;
    train)
        run_training "$@"
        ;;
    inference)
        run_inference "$@"
        ;;
    experiment)
        run_experiments
        ;;
    test)
        run_tests
        ;;
    augment)
        run_augmentation
        ;;
    ensemble)
        run_ensemble
        ;;
    help|"")
        show_help
        ;;
    *)
        print_message "Unknown command: $1" $RED
        show_help
        exit 1
        ;;
esac