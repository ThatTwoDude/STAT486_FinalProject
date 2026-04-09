# STAT486_FinalProject

# The Cinderella Detector: Predicting NCAA Upsets via Anomaly Detection

**STAT 486 Final Project Proposal**
**Authors:** Anton / Aaron / James
**Live Interactive Dashboard:** [Online Streamilt App](https://stat486final.streamlit.app/)

---

## Project Question, Hypothesis, and Motivation

### Motivation
Every year, the NCAA March Madness tournament captivates fans with unexpected "Cinderella" teams—low-seeded schools that upset tournament powerhouses. Traditionally, upset predictions rely heavily on official seeding, conference prestige, or basic matchup statistics. This project pivots away from those traditional markers to evaluate teams strictly on their "statistical DNA." 

By utilizing unsupervised anomaly detection on regular-season advanced metrics, we aim to identify under-seeded teams that possess the statistical profile of a powerhouse. Feeding these anomaly scores into a supervised classification model allows us to teach the model to look beyond a simple "12 vs. 5 seed" difference and recognize underlying elite performance profiles. 

### Project Question
Can unsupervised anomaly detection of regular-season advanced metrics successfully identify underlying elite performance profiles in low-seeded NCAA basketball teams, and does integrating these anomaly scores improve the predictive accuracy of supervised upset classification models during the NCAA Tournament?

### Hypothesis
We hypothesize that an Anomaly Score—generated via unsupervised learning (Isolation Forests or Gaussian Mixture Models) on regular-season metrics—will serve as a highly predictive feature for a supervised classification model (e.g., XGBoost, Logistic Regression). We expect models utilizing this integrated anomaly score to outperform models relying solely on standard matchup statistics and official seeding when predicting binary matchup outcomes (Win/Loss) for lower-seeded teams.

---

## Main Results and Insights

Our machine learning pipeline yielded three distinct phases of insights regarding the anatomy of NCAA upsets:

**EDA Insights**

* **The "Small Data" Reality:** True Cinderellas are incredibly rare. After definining a Cinderella as a 10+ seed that reaches the Sweet 16, we found only ~45 true Cinderellas in the modern tournament era. They represent a severe class imbalance

* **The "Expectation Gap":** We suspected that Cinderellas rarely look like standard 12-seeds. Before the tournament even begins, we expected underlying metrics heavily over-performed in their assigned seed line. 

**Supervised Modeling Results (XGBoost)**

We compared a Random Forest Baseline against a heavily regularized XGBoost model built to handle small, imbalanced data.

| **Model Type** | **Key Hyperparameters** | **Validation Setup** | **PR-AUC** |
| ------------------ | ------------------ | ------------------ | ------------------ |
| **Random Forest (Baseline)** | `max_depth`: 5 <br> `n_estimators`: 100 <br> `class_weight`: 'balanced' | Stratified 5-Fold CV | 0.1177 |
| **XGBoost Classifier** | `max_depth`: 2 <br> `learning_rate`: 0.01 <br> `colsample_bytree`: 0.5 <br> `gamma`: 0.1 | Stratified 5-Fold CV | 0.1571 |

* **Key Takeaway**: XGBoost successfully identified the majority of historical Cinderellas (XGBoost recall of 0.6679 compared to 0.2429 in Random Forest). Feature Importance and SHAP analysis revealed that `Rank_vs_tourneyAvg` and `SeedNum` were the primary drivers of predictions. The model ultimately learned to hunt for "Expectation Gaps"-attempting to flag low-seeded teams whose underlying regular-season efficiency metrics, like Massey Rank in Rank_vs_TourneyAvg, outpaced traditional expectations of their assigned low seed. 

![alt text](/assets/shap_summary.png)

**Unsupervised Modeling Results (Isolation Forest & GMM)**
* **The Anomaly Myth:** Our Isolation Forest proved our initial hypothesis wrong—Cinderellas are not statistical anomalies. They fall squarely within standard anomaly distributions.

* **The Archetype Confirmation:** However, our Gaussian Mixture Model (GMM) successfully grouped roughly 82% of historical Cinderellas into a single, specific archetype cluster. This proves Cinderellas don't rely on chaotic variance; they share a highly specific, repeatable baseline that mimics the statistical profile of higher-seeded teams.

![alt text](/assets/GMM2.png)
---

## Data Sources and Usage Rights

**Primary Data Source:** * [March Machine Learning Mania Dataset](https://www.kaggle.com/c/march-machine-learning-mania-2026/data) (Hosted on Kaggle): Provides decades of clean, highly structured regular-season and tournament box scores, advanced metrics, and play-by-play data.

**Ethical and Legal Considerations:**
There is no Personally Identifiable Information (PII) at risk, as the data only involves public figures (student-athletes and universities) and their on-court statistics. 

**License and Usage Notes:**
Per the Kaggle Competition Rules, this dataset is available for academic research and education. However, **we are strictly prohibited from redistributing the data to anyone who has not formally agreed to the competition rules.** To comply with these rules and maintain a lightweight repository, **no raw or processed data files are tracked in this GitHub repository.** Anyone wishing to reproduce this project must independently acquire the data via Kaggle using the instructions below.

---

## Steps to Reproduce 

To reproduce our results, you must first pull the required data locally. 

### 1. Environment Setup

Clone this repository and install the required dependencies using the `requirements.txt` file.

```bash
git clone git@github.com:ThatTwoDude/STAT486_FinalProject.git
cd STAT486_FinalProject
pip install -r requirements.txt
```

### 2. Data Retrieval Instructions

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
* We have provided a Python script to handle downloading and extracting the dataset programmatically, bypassing any system PATH issues. 

* Navigate to the `src/` directory in your terminal and run:
`python download_data.py`

* This will authenticate your Kaggle account, download the 2026 competition data, extract all CSVs into the `data/raw/` folder, and clean up the leftover zip file.

### 3. Run the Jupyter Notebooks

To strictly avoid data leakage and ensure proper preprocessing, the notebooks must be run in the following chronological order. *(Note: Restart the kernel and select "Run ALl" for each)*

1. `EDA.ipynb`: Cleans the raw Kaggle data, merges game-by-game stats into team-season aggregates, and handles early exploratory analysis. Located in the `STAT486_FinalProject/data/`

2. `03_supervised_modeling.ipynb`: Trains the baseline and XGBoost models, and exports the finalized `.pk1` models to the `/models` directory. Located in `STAT486_FinalProject/notebooks/`

3. `04_unsupervised_modeling.ipynb`: Explores the statistical archetypes using Isolation Forests and Gaussian Mixture Models. Located in `STAT486_FinalProject/notebooks/`

### 4. Run the Interactive Dashboard

Once the notebooks have successfully exported the models to the `models/` directory, you can launch the local interactive application to test hypothetical teams or backtest historical seasons.

```bash
streamlit run app.py
```