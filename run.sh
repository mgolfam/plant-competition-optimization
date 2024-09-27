#!/bin/bash

# Step 1: Create a virtual environment (optional but recommended)
# echo "Creating virtual environment..."
# python3 -m venv venv

# Step 2: Activate the virtual environment
echo "Activating virtual environment..."
conda activate

# Step 3: Install the dependencies
# echo "Installing dependencies..."
# pip install -r requirements.txt

# Step 4: Run the main Python script
echo "Running the main algorithm..."
python src/main.py

# Optional: Run tests (uncomment the line below if you want to run tests)
# echo "Running tests..."
# python3 -m unittest discover -s tests

# Deactivate the virtual environment after execution
# echo "Deactivating virtual environment..."
# deactivate
