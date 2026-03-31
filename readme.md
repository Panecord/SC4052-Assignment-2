# Pre-Requisites

Data folder that contains `web-Google.txt` (full) or `web-Google_10k.txt`.

# Create Python Virtual Environment

py -m venv .venv

# Activate Environment

.venv\Scripts\activate

# Install Pre-requisites

pip install -r requirements.txt

# Run Python File for Analysis Summary and Graph Generation

python pagerank_q6.py

# Optional: run the smaller 10k dataset for faster debugging

python pagerank_q6.py --dataset data/web-Google_10k.txt

# Generated graphs will be stored in the outputs/ folder
