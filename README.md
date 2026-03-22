# STAT486_FinalProject

# The Cinderella Detector: Predicting NCAA Upsets via Anomaly Detection

**STAT 486 Final Project Proposal**
**Authors:** Anton / Aaron / James

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

## Data Sources and Usage Rights

**Primary Data Source:** * [March Machine Learning Mania Dataset](https://www.kaggle.com/c/march-machine-learning-mania-2026/data) (Hosted on Kaggle): Provides decades of clean, highly structured regular-season and tournament box scores, advanced metrics, and play-by-play data.

**Ethical and Legal Considerations:**
There is no Personally Identifiable Information (PII) at risk, as the data only involves public figures (student-athletes and universities) and their on-court statistics. 

**License and Usage Notes:**
Per the Kaggle Competition Rules, this dataset is available for academic research and education. However, **we are strictly prohibited from redistributing the data to anyone who has not formally agreed to the competition rules.** To comply with these rules and maintain a lightweight repository, **no raw or processed data files are tracked in this GitHub repository.** Anyone wishing to reproduce this project must independently acquire the data via Kaggle using the instructions below.

---

## Steps to Reproduce 

To reproduce our results, you must first pull the required data locally. 

### 1. Data Retrieval Instructions
1. **Gain Kaggle Access:** Create an account at [kaggle.com](https://www.kaggle.com/). Navigate to the current March Madness competition and click **Join Competition** (requires phone multi-factor authentication).
2. **Install Kaggle CLI:** Ensure the Kaggle package is installed in your Python environment: 
   ```bash
   pip install kaggle
