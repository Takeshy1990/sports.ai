# 🏀 Sports Match Outcome Prediction — Euroleague Basketball

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/ML-ScikitLearn-orange)
![Matplotlib](https://img.shields.io/badge/Plots-Matplotlib-green)
![ReportLab](https://img.shields.io/badge/PDF-ReportLab-red)

End-to-end **Machine Learning pipeline** that predicts:

- **Home win probability** (classification)
- **Total points scored** (regression, for Over/Under analysis)

The project is designed to showcase **data cleaning, feature engineering, model training, evaluation, visualization, automated reporting**, and **actionable insights** for sports analytics.

---

## 📊 Data

- **Historical matches** → `matches.csv`  
  Contains past Euroleague games with pre-match statistics + final scores.  

- **Upcoming fixtures** → `fixtures.csv`  
  Games without results, used for prediction.

### Features
- Team offensive/defensive stats (PPG, OPPG, FG%, 3P%, FT%).  
- Rebounds, assists, turnovers, possessions.  
- Contextual factors: win/loss streaks, head-to-head winrate, days of rest, match importance.  
- Target variables:
  - `win_home` (1 if home team wins, else 0)  
  - `total_points` (home_points + away_points)

---

## ⚙️ Pipeline Overview

1. **Training (`sports_train.py`)**
   - Builds engineered features (home–away differentials).  
   - Trains:
     - Gradient Boosting Classifier → home win probability  
     - Gradient Boosting Regressor → total points  
   - Outputs performance metrics and plots:
     - `roc_curve.png`
     - `feature_importance.png`
     - `reg_scatter.png`

2. **Prediction (`sports_predict.py`)**
   - Loads saved models.  
   - Predicts win probabilities & total points for `fixtures.csv`.  
   - Optionally compares to an Over/Under line (`--line 160.5`).  
   - Saves `prediction.csv`.

3. **Reporting (`make_report.py`)**
   - Generates `sports_report.pdf` with metrics, plots, and predictions.  
   - Executive summary with key insights.

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn matplotlib joblib reportlab

# 2. Train models
python sports_train.py

# 3. Make predictions (with optional Over/Under line)
python sports_predict.py --line 160.5

# 4. Generate PDF report
python make_report.py
Outputs:

sports_clf.pkl, sports_reg.pkl — trained models

sports_meta.json — performance metrics

prediction.csv — predictions on fixtures

sports_report.pdf — final report with results

📈 Results (Demo)
Model performance (on hold-out test set):

Classification Accuracy: ~0.72

ROC–AUC: ~0.78

Regression MAE: ~7.5 points

Regression RMSE: ~10.2 points

Feature importance:
Top drivers include:

Field Goal % differential

Turnover differential

Points per game (PPG) differential

Example predictions (prediction.csv):

date	home_team	away_team	win_prob_home	win_prob_away	pred_total_points	suggestion
2025-04-12	Real Madrid	Panathinaikos	60.3%	39.7%	161.2	Home ML · Over 160.5
2025-04-13	Olympiacos	Fenerbahce	55.1%	44.9%	164.0	Home ML · Over 160.5

📊 Example Plots
ROC Curve (Classifier)


Feature Importance (Classifier)


Actual vs Predicted (Regressor)


📑 Executive Summary
The model successfully predicts win probabilities and total points for Euroleague games.

FG%, Turnovers, and Scoring differentials are key drivers of outcomes.

Predictions can support sports analytics and betting models (value bet detection).

This project demonstrates skills in:

Feature engineering

Supervised learning (classification + regression)

Model evaluation

Automated reporting (PDF)

End-to-end data pipeline development