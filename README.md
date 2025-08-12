# Microsoft-Azure-Predictive-Maintenance

This an end-to-end ML solution for the prediction of machine failures. This code is based on a predictive maintenance project using Azure and Kaggle datasets. It downloads datasets, processes them, and prepares them for analysis.
It includes data preprocessing, exploratory data analysis (EDA), and model creating/training scripts. The project is structured to facilitate easy understanding and modification for predictive maintenance tasks.

This project runs on Python 3.13 and uses a virtual environment for package management and isolation made with Astral's UV.
For accurate running, ensure you have both Python 3.13 and the `uv` package installed. UV can be installed via pip:

```bash
pip install uv
```
After installing `uv`, you can create a virtual environment and activate it with the following commands:

```bash
uv sync
source .venv/bin/activate
```

Run the scripts in the following order:
1. **Data Acquisition**: `src/data_acquisition.py`
   - Downloads and saves datasets from Kaggle Storage.
   
**(Optional: Exploratory Data Analysis (EDA))** `src/eda.py`
   - Performs exploratory data analysis (EDA) on the datasets.
2. **Data Preprocessing**: `src/data_preprocessing.py`
   - Cleans and prepares the data for analysis.
3. **Model**: `src/model.py`
   - Trains and evaluates the machine learning model on the preprocessed data.
4. **Deployment**: `src/app.py`
   - Deploys the trained model as a REST API using FastAPI.