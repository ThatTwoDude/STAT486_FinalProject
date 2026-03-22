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
* You must click on Join Competition, it will require you to enter in your phone number for multi-factor authentication, but after that you will have access to the data

### 2. Setup Kaggle API Authentication
You will need a Kaggle account and an API token to download the data programmatically.
* Go to your Kaggle account settings and click **"Create New Token"** to download a `kaggle.json` file.
* Place the `kaggle.json` file in your local `~/.kaggle/` directory (Mac/Linux) or `C:\Users\<Windows-username>\.kaggle\` (Windows).
* Secure the file permissions (Mac/Linux only): `chmod 600 ~/.kaggle/kaggle.json`

### 3. Install the Kaggle CLI
Ensure the Kaggle package is installed in your Python environment:
`pip install kaggle`

### 4. Download the Dataset
Navigate to the root of this project in your terminal and run the following commands to download and extract the dataset directly into the `data/raw/` folder:

`cd data`
`kaggle competitions download -c march-machine-learning-mania-2026` 
`unzip ncaa-basketball-dataset.zip -d raw/`

*(Note: If using a specific year's competition data, replace the dataset slug in the download command with the appropriate Kaggle competition identifier).*

## Directory Structure
* `raw/`: Unaltered, original CSV files downloaded directly from the source. **Do not manually edit these files.**
* `processed/`: Cleaned data, feature-engineered datasets, and merged files ready for modeling.