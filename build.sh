#!/usr/bin/env bash
# This script runs during the build phase

echo "Running custom build steps..."

# Install Python packages (optional if you have requirements.txt)
pip install -r requirements.txt

# Create folders or move files if needed
mkdir -p instance

echo "Build script completed."
