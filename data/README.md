# Data Repository: The Cinderella Detector

This folder contains the raw and processed datasets required for the Cinderella Detector machine learning pipeline. 

## Primary Data Source
The primary dataset for this project is the **March Machine Learning Mania** dataset, hosted on Kaggle. This dataset includes decades of historical NCAA regular-season and tournament data.

## Data Rules

The Kaggle Competition has specific rules on who can access the data, which are listed below:

4. COMPETITION DATA
a. Data Access and Use.

You may access and use the Competition Data for any purpose, whether commercial or non-commercial, including for participating in the Competition and on Kaggle.com forums, and for academic research and education. The Competition Sponsor reserves the right to disqualify any Participant who uses the Competition Data other than as permitted by the Competition Website and these Rules.
b. Data Security.

You agree to use reasonable and suitable measures to prevent persons who have not formally agreed to these Rules from gaining access to the Competition Data. You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition. You agree to notify Kaggle immediately upon learning of any possible unauthorized transmission of or unauthorized access to the Competition Data and agree to work with Kaggle to rectify any unauthorized transmission or access.

## Data Retrieval Instructions

To maintain reproducibility and keep the repository lightweight, raw datasets are not tracked in version control. Follow these steps to retrieve the data locally:

### 1. Gaining Access to the Data
You will need a Kaggle account in order to access the data. 
* You can create an account at [kaggle.com](https://www.kaggle.com). 
* Then you must navigate to the current [march madness competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data)
* You must click on Join Competition, it will require you to enter in your phone number for multi-factor authentication, but after that you will have access to the data.
* You can else access the data here by navigating to the data tab and downloading all, or download it programatically through the following steps

### 2. Install the Kaggle CLI
Ensure the Kaggle package is installed in your Python environment:
`pip install kaggle`

### 4. Setup Kaggle API Authentication
You will need a Kaggle account and an API token to download the data programmatically.
* Go to your Kaggle [account settings](https://www.kaggle.com/settings) and click **"Create New Token"** under the **"API"** section.
* Create it as an environment variable, by running the following command:
`export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx # Copied from the settings UI`

### 4. Download the Dataset
We have provided a Python script to handle downloading and extracting the dataset programmatically, bypassing any system PATH issues. 

Navigate to the `src/` directory in your terminal and run:
`python download_data.py`

This will authenticate your Kaggle account, download the 2026 competition data, extract all CSVs into the `data/raw/` folder, and clean up the leftover zip file.

## Directory Structure
* `raw/`: Unaltered, original CSV files downloaded directly from the source. **Do not manually edit these files.**
* `processed/`: Cleaned data, feature-engineered datasets, and merged files ready for modeling.