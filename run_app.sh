#!/bin/bash

# Set environment variables for better performance
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings
export OMP_NUM_THREADS=1             # Optimize for single-threaded inference

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

echo "Initialize Database..."
python db-init.py

# Run the Streamlit app
echo "Starting the Ad Recommender App..."
streamlit run app.py --server.port 8501 --server.address localhost
