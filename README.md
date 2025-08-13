# SurgiSense: Post-Operative Complication Risk Predictor

## Overview
SurgiSense is an open-source AI project designed to predict post-operative complications using interpretable machine learning. Built with a focus on transparency, clinical relevance, and ease of deployment, SurgiSense aims to support clinicians in real-time surgical risk assessment.

## Roadmap
Check out the [MVP Roadmap Issue](https://github.com/Shlokbansal/surgisense/issues/1) to see current progress and upcoming milestones.

## Project Status

- COMPLETE: Data loader module complete (`load_data.py`)
- COMPLETE: Preprocessing utilities implemented (`utils.py`)
- COMPLETE: Dataset exploration + EDA complete (`eda.ipynb`)
- COMPLETE: First model trained using Random Forest + SHAP interpretability (`eda_complete.ipynb`)
- IN PROGRESS: Additional model prototyping in progress (XGBoost, LightGBM)
- ⬜ Streamlit-based demo interface (in development)

## Project Structure
surgisense/
│
├── data/ # Raw and processed data files
├── notebooks/ # Jupyter notebooks for EDA and prototyping
├── src/ # Source code (e.g., model training, utils)
├── models/ # Saved models (excluded from Git)
├── reports/ # Generated reports and visualizations
├── sql/ # Any SQL queries or database scripts
├── requirements.txt # Python dependencies
└── README.md # Project overview and usage

## Goals
- Simulate or collect real-world post-operative patient data
- Train classification models to predict complications
- Evaluate performance (e.g., AUC, precision, recall)
- Provide explainability using SHAP or similar tools
- Deploy as an API or Streamlit app for demos

## Dependencies
See `requirements.txt` for the list of Python packages.

## Setup Instructions
1. Clone the repo  
2. Set up virtual environment  
3. Install dependencies with `pip install -r requirements.txt`  
4. Explore `notebooks/` for EDA and modeling work

## Contributors
- Shlok Bansal (Owner)

## License
MIT License (or choose one appropriate)

