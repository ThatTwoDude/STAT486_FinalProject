# STAT486_FinalProject

# The Cinderella Detector: Predicting NCAA Upsets via Anomaly Detection

**STAT 486 Final Project Proposal**
**Authors:** Anton / Aaron / James

---

## Project Question, Hypothesis, and Motivation

### Motivation
Every year, the NCAA March Madness tournament captivates fans with unexpected "Cinderella" teams—low-seeded schools that upset tournament powerhouses. Traditionally, upset predictions rely heavily on official seeding, conference prestige, or basic matchup statistics. This project pivots away from those traditional markers to evaluate teams strictly on their "statistical DNA." 

By utilizing unsupervised anomaly detection on regular-season advanced metrics, we aim to identify under-seeded teams that possess the statistical profile of a powerhouse. Feeding these anomaly scores into a supervised classification model allows us to teach the model to look beyond a simple "12 vs. 5 seed" difference and recognize underlying elite performance profiles. 

*Disclaimer: This project is for academic demonstration and educational purposes only. It is not intended, nor endorsed, for financial sports betting.*

### Project Question
Can unsupervised anomaly detection of regular-season advanced metrics successfully identify underlying elite performance profiles in low-seeded NCAA basketball teams, and does integrating these anomaly scores improve the predictive accuracy of supervised upset classification models during the NCAA Tournament?

### Hypothesis
We hypothesize that an Anomaly Score—generated via unsupervised learning (Isolation Forests or Gaussian Mixture Models) on regular-season metrics—will serve as a highly predictive feature for a supervised classification model (e.g., XGBoost, Logistic Regression). We expect models utilizing this integrated anomaly score to outperform models relying solely on standard matchup statistics and official seeding when predicting binary matchup outcomes (Win/Loss) for lower-seeded teams.

---

## Data Sources and Usage Rights

**Primary Data Source:** * [March Machine Learning Mania Dataset](https://www.kaggle.com/c/march-machine-learning-mania-2026/data) (Hosted on Kaggle): Provides decades of clean, highly structured regular-season and tournament box scores, advanced metrics, and play-by-play data.

**Ethical and Legal Considerations:**
There is no Personally Identifiable Information (PII) at risk, as the data only involves public figures (student-athletes and universities) and their on-court statistics. 

**License and Usage Notes (CRITICAL):**
Per the Kaggle Competition Rules, this dataset is available for academic research and education. However, **we are strictly prohibited from redistributing the data to anyone who has not formally agreed to the competition rules.** To comply with these rules and maintain a lightweight repository, **no raw or processed data files are tracked in this GitHub repository.** Anyone wishing to reproduce this project must independently acquire the data via Kaggle using the instructions below.

---

## Steps to Reproduce 

To reproduce our results, you must first pull the required data locally. 

### 1. Data Retrieval Instructions

1. Gaining Access to the Data
You will need a Kaggle account in order to access the data. 
* You can create an account at [kaggle.com](https://www.kaggle.com). 
* Then you must navigate to the current [march madness competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data)
* You must click on Join Competition, it will require you to enter in your phone number for multi-factor authentication, but after that you will have access to the data.
* You can else access the data here by navigating to the data tab and downloading all, or download it programatically through the following steps

2. Install the Kaggle CLI
Ensure the Kaggle package is installed in your Python environment:
`pip install kaggle`

3. Setup Kaggle API Authentication
You will need a Kaggle account and an API token to download the data programmatically.
* Go to your Kaggle [account settings](https://www.kaggle.com/settings) and click **"Create New Token"** under the **"API"** section.
* Create it as an environment variable, by running the following command:
`export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx # Copied from the settings UI`

4. Download the Dataset
We have provided a Python script to handle downloading and extracting the dataset programmatically, bypassing any system PATH issues. 

Navigate to the `src/` directory in your terminal and run:
`python download_data.py`

This will authenticate your Kaggle account, download the 2026 competition data, extract all CSVs into the `data/raw/` folder, and clean up the leftover zip file.