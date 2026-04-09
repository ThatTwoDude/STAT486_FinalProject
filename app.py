import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="The Cinderella Detector", page_icon="🏀", layout="wide")

# --- DATA & MODEL LOADING ---
@st.cache_resource
def load_models():
    """Loads the trained models and scaler from the /models directory."""
    try:
        xgb_model = joblib.load('models/xgboost_model.pkl')
        iso_forest = joblib.load('models/isolation_forest.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return xgb_model, iso_forest, scaler
    except Exception as e:
        st.error(f"⚠️ Could not load models. Error: {e}")
        return None, None, None

@st.cache_data
def load_historical_data():
    """Loads the raw metrics data for backtesting."""
    try:
        df = pd.read_csv('data/metrics.csv')
        df = df.dropna(subset=['SeedNum', 'AvgScoreDiff', 'MasseyRankMean']).copy()
        return df
    except Exception as e:
        st.error(f"⚠️ Could not load metrics.csv. Make sure it is in the data/ folder. Error: {e}")
        return None

xgb_model, iso_forest, scaler = load_models()
historical_df = load_historical_data()

# Features specifically used by the Isolation Forest
stats_for_anomaly = [
    'AvgScoreDiff', 'WinPct', 'ScoreDiff_vs_TourneyAvg', 'Rank_vs_TourneyAvg', 
    '3PPct_vs_TourneyAvg', 'FGPct_vs_TourneyAvg', 'AstTO_vs_TourneyAvg', 'Steals_vs_TourneyAvg'
]

# OPTIMAL THRESHOLD (From Notebook Cell 18)
OPTIMAL_THRESHOLD_DECIMAL = 0.5009
OPTIMAL_THRESHOLD_PCT = 50.09

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🏀 Cinderella Detector")
app_mode = st.sidebar.radio("Select a Tool:", ["🔮 Single Team Predictor", "📊 Historical Season Tester"])
st.sidebar.markdown("---")

# ==========================================
# MODE 1: SINGLE TEAM PREDICTOR
# ==========================================
if app_mode == "🔮 Single Team Predictor":
    st.title("🔮 Single Team Predictor")
    st.markdown("Enter a low-seeded candidate's regular-season metrics to calculate their Anomaly Score and Sweet 16 Probability.")
    
    # Put inputs in the sidebar ONLY for this mode
    st.sidebar.header("Team Profile Input")
    
    st.sidebar.subheader("General Stats")
    seed_num = st.sidebar.slider("Tournament Seed", 10, 16, 12)
    win_pct = st.sidebar.slider("Win Percentage", 0.400, 0.900, 0.700, format="%.3f")
    avg_score_diff = st.sidebar.number_input("Average Score Differential", value=5.5, step=0.5)
    massey_variance = st.sidebar.number_input("Massey Rank Variance (Mean - Median)", value=2.0, step=0.5)

    st.sidebar.subheader("Expectation Gaps")
    score_diff_gap = st.sidebar.number_input("Score Diff vs. Tourney Avg", value=0.5, step=0.5)
    rank_gap = st.sidebar.number_input("Rank vs. Tourney Avg", value=-40.0, step=1.0)
    three_pt_gap = st.sidebar.number_input("3P% vs. Tourney Avg", value=0.015, step=0.005, format="%.3f")
    fg_pt_gap = st.sidebar.number_input("FG% vs. Tourney Avg", value=-0.010, step=0.005, format="%.3f")
    ast_to_gap = st.sidebar.number_input("Ast/TO Ratio vs. Tourney Avg", value=0.10, step=0.05)
    steals_gap = st.sidebar.number_input("Steals vs. Tourney Avg", value=1.5, step=0.1)

    input_dict = {
        'SeedNum': seed_num, 'AvgScoreDiff': avg_score_diff, 'WinPct': win_pct,
        'MasseyVariance': massey_variance, 'ScoreDiff_vs_TourneyAvg': score_diff_gap,
        'Rank_vs_TourneyAvg': rank_gap, '3PPct_vs_TourneyAvg': three_pt_gap,
        'FGPct_vs_TourneyAvg': fg_pt_gap, 'AstTO_vs_TourneyAvg': ast_to_gap,
        'Steals_vs_TourneyAvg': steals_gap
    }

    if st.button("🔍 Run Cinderella Detector", type="primary"):
        if xgb_model and iso_forest and scaler:
            input_df = pd.DataFrame([input_dict])
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            # --- PHASE 1: ANOMALY DETECTION ---
            iso_score = iso_forest.decision_function(input_df[stats_for_anomaly])[0]
            input_df['Isolation_Score'] = iso_score
            
            with col1:
                st.subheader("Phase 1: Statistical Profile")
                st.metric(label="Unsupervised Isolation Score", value=round(iso_score, 4))
                if iso_score < 0:
                    st.success("🚨 **High Anomaly Detected!** This team's efficiency deviates heavily from a standard underdog.")
                else:
                    st.info("📊 **Standard Profile.** This team fits the traditional distribution for their seed line.")

            # --- PHASE 2: XGBOOST PREDICTION ---
            features_order = list(input_dict.keys()) + ['Isolation_Score']
            input_scaled = scaler.transform(input_df[features_order])
            prob = xgb_model.predict_proba(input_scaled)[0][1] * 100
            
            with col2:
                st.subheader("Phase 2: Cinderella Probability")
                st.metric(label="Probability of Sweet 16 Run", value=f"{prob:.1f}%")
                if prob >= OPTIMAL_THRESHOLD_PCT:
                    st.balloons()
                    st.success("🎯 **PREDICTION: CINDERELLA ALERT**")
                    st.write("This team possesses the necessary Expectation Gaps to pull off a deep tournament run.")
                else:
                    st.warning("📉 **PREDICTION: EARLY EXIT**")
                    st.write(f"This team falls below our strict prediction threshold ({OPTIMAL_THRESHOLD_PCT}%).")
            
            # --- INTERPRETABILITY SECTION ---
            st.markdown("---")
            st.subheader("Model Interpretability")
            st.write("Why did the model make this decision? According to our SHAP and Feature Importance analysis, the model heavily weighs **Rank vs. Tourney Avg** and **Seed Number**.")
            
            if os.path.exists('assets/shap_summary.png'):
                st.image('assets/shap_summary.png', caption='Global Feature Importance (SHAP Summary)')
            else:
                st.info("Tip: Save your SHAP summary plot to `assets/shap_summary.png` to display it here.")

# ==========================================
# MODE 2: HISTORICAL SEASON TESTER
# ==========================================
elif app_mode == "📊 Historical Season Tester":
    st.title("📊 Historical Season Tester")
    st.markdown("Select a past NCAA tournament season to see which teams our model would have flagged as Cinderellas, and whether they actually pulled it off.")
    
    if historical_df is not None:
        available_seasons = sorted(historical_df['Season'].unique(), reverse=True)
        selected_season = st.selectbox("Select a Tournament Season:", available_seasons)
        
        if st.button("Run Historical Analysis", type="primary"):
            # 1. Filter to the selected season
            season_df = historical_df[historical_df['Season'] == selected_season].copy()
            
            # 2. Recreate the Expectation Gaps for this specific season
            season_df['FG_Pct'] = season_df['AvgFGM'] / season_df['AvgFGA']            
            season_df['3P_Pct'] = season_df['AvgFGM3'] / season_df['AvgFGA3']          
            season_df['Ast_TO_Ratio'] = season_df['AvgAssists'] / season_df['AvgTurnovers']
            season_df['MasseyVariance'] = season_df['MasseyRankMean'] - season_df['MasseyRankMedian']
            
            tourney_avg = season_df[['AvgScoreDiff', 'MasseyRankMean', 'AvgPointsFor', 'FG_Pct', '3P_Pct', 'Ast_TO_Ratio', 'AvgSteals']].mean()
            
            season_df['ScoreDiff_vs_TourneyAvg'] = season_df['AvgScoreDiff'] - tourney_avg['AvgScoreDiff']
            season_df['Rank_vs_TourneyAvg'] = tourney_avg['MasseyRankMean'] - season_df['MasseyRankMean']
            season_df['3PPct_vs_TourneyAvg'] = season_df['3P_Pct'] - tourney_avg['3P_Pct']
            season_df['FGPct_vs_TourneyAvg'] = season_df['FG_Pct'] - tourney_avg['FG_Pct']
            season_df['AstTO_vs_TourneyAvg'] = season_df['Ast_TO_Ratio'] - tourney_avg['Ast_TO_Ratio']
            season_df['Steals_vs_TourneyAvg'] = season_df['AvgSteals'] - tourney_avg['AvgSteals']
            
            # 3. Filter to Underdog Candidates (Seeds 10-16)
            candidates = season_df[season_df['SeedNum'] >= 10].copy()
            
            # EXACT BUG FIX: Use MaxTourneyDayNum >= 143 to properly identify Sweet 16 teams
            candidates['Actual_Cinderella'] = ((candidates['SeedNum'] >= 10) & (candidates['MaxTourneyDayNum'] >= 143)).astype(int)
            
            if len(candidates) > 0 and xgb_model and iso_forest and scaler:
                st.markdown("---")
                
                # Calculate Isolation Score
                candidates['Isolation_Score'] = iso_forest.decision_function(candidates[stats_for_anomaly])
                
                # Setup Features for XGBoost
                base_features = [
                    'SeedNum', 'AvgScoreDiff', 'WinPct', 'MasseyVariance',
                    'ScoreDiff_vs_TourneyAvg', 'Rank_vs_TourneyAvg', '3PPct_vs_TourneyAvg',
                    'FGPct_vs_TourneyAvg', 'AstTO_vs_TourneyAvg', 'Steals_vs_TourneyAvg'
                ]
                features_order = base_features + ['Isolation_Score']
                
                # Scale and Predict
                X_scaled = scaler.transform(candidates[features_order])
                probs = xgb_model.predict_proba(X_scaled)[:, 1]
                candidates['Predicted_Probability'] = probs
                
                # Apply our optimally tuned threshold
                candidates['Model_Prediction'] = (candidates['Predicted_Probability'] >= OPTIMAL_THRESHOLD_DECIMAL).astype(int)
                
                # Calculate simple metrics for the UI using the unformatted 1s and 0s (Fixes the Pandas mapping bug)
                true_positives = len(candidates[(candidates['Model_Prediction'] == 1) & (candidates['Actual_Cinderella'] == 1)])
                actual_cinderellas = len(candidates[candidates['Actual_Cinderella'] == 1])
                predicted_cinderellas = len(candidates[candidates['Model_Prediction'] == 1])
                
                # Format a clean results table for display
                results_table = candidates[['TeamName', 'SeedNum', 'Predicted_Probability', 'Model_Prediction', 'Actual_Cinderella']].sort_values(by='Predicted_Probability', ascending=False)
                results_table['Predicted_Probability'] = (results_table['Predicted_Probability'] * 100).apply(lambda x: f"{x:.1f}%")
                results_table['Model_Prediction'] = results_table['Model_Prediction'].map({1: 'Yes', 0: 'No'})
                results_table['Actual_Cinderella'] = results_table['Actual_Cinderella'].map({1: '✅ Yes', 0: '❌ No'})
                
                # UI Display
                st.markdown("### 🏆 Tournament Post-Mortem")
                col1, col2, col3 = st.columns(3)
                col1.metric("Actual Cinderellas", actual_cinderellas)
                col2.metric("Model Flags (Alarms)", predicted_cinderellas)
                
                if actual_cinderellas > 0:
                    recall = (true_positives / actual_cinderellas) * 100
                    col3.metric("Recall (Caught %)", f"{recall:.1f}%")
                else:
                    col3.metric("Recall (Caught %)", "N/A (No Cinderellas)")
                
                st.markdown("#### Candidate Threat Leaderboard")
                st.dataframe(results_table, use_container_width=True, hide_index=True)
            else:
                st.warning("Not enough data to run analysis for this season.")