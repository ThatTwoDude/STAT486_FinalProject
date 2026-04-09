# Senior Machine Learning Project Proposal
**Project Title:** The Cinderella Detector: Predicting NCAA Upsets via Anomaly Detection

## Candidate Project Ideas Generated with AI
1. **The Cinderella Detector (Selected):** Using unsupervised anomaly detection on regular-season advanced metrics to identify under-seeded NCAA basketball teams, and feeding those anomaly scores into a supervised classification model to predict tournament upsets.
2. **Redefining Basketball Positions:** Applying K-Means clustering to NBA player tracking data to discover modern, data-driven "player archetypes," and then using those archetypes in a regression model to predict future player market value or Win Shares.
3. **Strategic Shot Optimization:** Building a contextual recommender system to determine a player's optimal shot zone based on defensive alignment, paired with a supervised model to calculate the Expected Effective Field Goal Percentage of the shots they actually take.

## AI Influence on Idea Selection
Brainstorming with the AI helped pivot the project from a broad basketball related project to a specific NCAA Men's Basketball project. Initially, we just planned to do something with basketball, and we had found the NCAA March Madness Tournament data, and wanted to do something with that. The AI suggested a more specific and concrete approach, where the output of an unsupervised model (anomaly detection) directly feeds into a supervised model to predict the most likely to win cinderella teams. Additionally, it allowed us to incoporate an unsupervised model into our original plan.

## AI Exchange Excerpt
> **User:** I want to look at the Cinderella Detector idea. Can you expand more on the project idea and how it would use both supervised learning (e.g., regression, classification) and one other ML technique, such as clustering, anomaly detection, recommender systems, or reinforcement learning. 
> 
> **AI:** "To make this a standout project, we need to ensure the two machine learning techniques aren't just sitting next to each other, but are actively working together in a unified pipeline... Your first step is to ignore the teams' names, conferences, and official seeds entirely. You will use unsupervised learning (Anomaly Detection) to evaluate them strictly on their statistical DNA... The Critical Integration: You will use the output from Phase 1 as a brand-new feature in Phase 2. You feed the supervised model the standard matchup stats, plus the 'Anomaly Score'... This teaches your classification model: 'Don't just look at the 12 vs. 5 seed difference. Notice that this 12-seed has the statistical anomaly score of a powerhouse.'"

## Final Research Question
Can unsupervised anomaly detection of regular-season advanced metrics successfully identify underlying elite performance profiles in low-seeded NCAA basketball teams, and does integrating these anomaly scores improve the predictive accuracy of supervised upset classification models during the NCAA Tournament?

## Candidate Target Variable for Supervised Analysis
The target variable is the **Binary Matchup Outcome (Win = 1, Loss = 0)** for the lower-seeded team in a given historical NCAA tournament game.

## Dataset Choice and Backup Dataset
* **Dataset:** Kaggle's "March Machine Learning Mania" dataset. It provides decades of clean, highly structured regular-season and tournament box scores, advanced metrics, and play-by-play data.

## Feasibility (Time, Compute, Scope)
This project is highly feasible for a single-semester timeline. The Kaggle dataset is pre-cleaned and tabular, minimizing the time spent on basic data wrangling and allowing more time for complex feature engineering. Because the data is tabular and text data, standard machine learning libraries in Python (`scikit-learn` and `xgboost`) will run efficiently on a standard personal laptop. The scope of .

## Ethical and Legal Considerations
The primary dataset is open-source and publicly available via Kaggle for educational and research purposes. There is no Personally Identifiable Information (PII) at risk, as the data only involves public figures (student-athletes and universities) and their on-court statistics. A standard disclaimer will be included in the final report stating that the model is for academic demonstration only and is not intended or endorsed for financial sports betting. Additionally, the only legal concern we have is that the competition requires us to not redistribute the data to anyone who has not agreed to the competition rules, so we will not place the data directly into out github repository.

## Planned Additional ML Methods
The primary supervised technique will be **Classification (e.g., Logistic Regression, Random Forest, or XGBoost)**. The planned additional, non-supervised technique is **Anomaly Detection (specifically Isolation Forests)** to calculate statistical deviation scores from the regular season data. If Isolation Forests do not yield distinct enough variance, **Gaussian Mixture Models (Clustering)** will be used as a fallback to group teams into distinct archetypal profiles.
