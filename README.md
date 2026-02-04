# Knowledge Garden IPD

A Streamlit prototype for a knowledge garden dashboard that estimates document shelf-life and highlights pruning candidates.

## Features
- Interactive dashboard with filters, pruning insights, and deep-dive explainability
- ArXiv dataset loader with transformation to the Knowledge Garden schema
- Sample Excel dataset for quick demos

## Project Structure
- Source/  Python app and data scripts
- Docs/    Requirements, notes, and diagrams
- Dataset/ Generated or collected data files

## Requirements
- Python 3.10+
- pip install streamlit pandas plotly openpyxl

## How To Run
1. From the repo root, run:
   streamlit run Source/app.py

2. Optional: regenerate the demo dataset:
   python Source/generate_data.py

3. Optional: refresh ArXiv column metadata:
   python Source/explore_kaggle_dataset.py
