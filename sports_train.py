# sports_train.py — Train classification (home win probability) + regression (total points)
# Inputs: matches.csv
# Outputs: sports_clf.pkl, sports_reg.pkl, sports_meta.json, roc_curve.png, reg_scatter.png, feature_importance.png

from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import joblib

BASE = Path(".").resolve()
MATCHES = BASE/"matches.csv"
CLF_MODEL = BASE/"sports_clf.pkl"
REG_MODEL = BASE/"sports_reg.pkl"
META_FILE = BASE/"sports_meta.json"
ROC_PNG   = BASE/"roc_curve.png"
REG_PNG   = BASE/"reg_scatter.png"
IMP_PNG   = BASE/"feature_importance.png"

def load_csv_any(p: Path) -> pd.DataFrame:
    for sep in (None, ",", ";", "\t"):
        try:
            return pd.read_csv(p, sep=sep, engine="python")
        except Exception:
            pass
    raise FileNotFoundError(f"Cannot read {p}")

def build_features(df: pd.DataFrame):
    df = df.copy()

    # Targets
    df["win_home"] = (df["home_points"] > df["away_points"]).astype(int)
    df["total_points"] = df["home_points"] + df["away_points"]

    # Candidate numeric columns (pre-match stats)
    base_num = [
        "home_ppg","away_ppg","home_oppg","away_oppg",
        "home_fg_pct","away_fg_pct","home_3p_pct","away_3p_pct","home_ft_pct","away_ft_pct",
        "home_reb","away_reb","home_ast","away_ast","home_tov","away_tov",
        "home_poss","away_poss",
        "home_streak","away_streak",
        "home_h2h_winrate","away_h2h_winrate",
        "home_days_rest","away_days_rest",
        "importance"
    ]

    # Differences (home - away) to capture match-up edges
    df["ppg_diff"]   = df["home_ppg"]   - df["away_ppg"]
    df["oppg_diff"]  = df["home_oppg"]  - df["away_oppg"]
    df["fg_diff"]    = df["home_fg_pct"]- df["away_fg_pct"]
    df["p3_diff"]    = df["home_3p_pct"]- df["away_3p_pct"]
    df["ft_diff"]    = df["home_ft_pct"]- df["away_ft_pct"]
    df["reb_diff"]   = df["home_reb"]   - df["away_reb"]
    df["ast_diff"]   = df["home_ast"]   - df["away_ast"]
    df["tov_diff"]   = df["home_tov"]   - df["away_tov"]
    df["poss_diff"]  = df["home_poss"]  - df["away_poss"]
    df["streak_diff"]= df["home_streak"]- df["away_streak"]
    df["h2h_diff"]   = df["home_h2h_winrate"] - df["away_h2h_winrate"]
    df["rest_diff"]  = df["home_days_rest"] - df["away_days_rest"]

    diff_cols = [
        "ppg_diff","oppg_diff","fg_diff","p3_diff","ft_diff",
        "reb_diff","ast_diff","tov_diff","poss_diff","streak_diff","h2h_diff","rest_diff",
        "importance"
    ]

    # final feature sets
    X_clf = df[diff_cols].copy()
    y_clf = df["win_home"].values

    X_reg = df[diff_cols].copy()
    y_reg = df["total_points"].values

    return X_clf, y_clf, X_reg, y_reg, diff_cols

def main():
    df = load_csv_any(MATCHES)

    X_clf, y_clf, X_reg, y_reg, feat_names = build_features(df)

    num_cols = X_clf.columns.tolist()
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False))
    ])

    pre = ColumnTransformer([("num", num_pipe, num_cols)], remainder="drop")

    # Models
    clf = GradientBoostingClassifier(random_state=42, n_estimators=400, learning_rate=0.05, max_depth=3)
    reg = GradientBoostingRegressor(random_state=42, n_estimators=500, learning_rate=0.05, max_depth=3)

    clf_pipe = Pipeline([("pre", pre), ("mdl", clf)])
    reg_pipe = Pipeline([("pre", pre), ("mdl", reg)])

    # Train/test split
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

    # Fit
    clf_pipe.fit(Xc_tr, yc_tr)
    reg_pipe.fit(Xr_tr, yr_tr)

    # Evaluate classification
    proba = clf_pipe.predict_proba(Xc_te)[:,1]
    yhat_c = (proba >= 0.5).astype(int)
    acc = accuracy_score(yc_te, yhat_c)
    try:
        auc = roc_auc_score(yc_te, proba)
    except ValueError:
        auc = float("nan")

    fpr, tpr, _ = roc_curve(yc_te, proba)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"GBM (AUC={auc:.3f})")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Home Win Classifier")
    plt.legend()
    plt.tight_layout(); plt.savefig(ROC_PNG); plt.close()

    # Evaluate regression
    yhat_r = reg_pipe.predict(Xr_te)
    mae = mean_absolute_error(yr_te, yhat_r)
    rmse = math.sqrt(mean_squared_error(yr_te, yhat_r))

    plt.figure(figsize=(5,5))
    plt.scatter(yr_te, yhat_r, alpha=0.6)
    mn, mx = min(yr_te.min(), yhat_r.min()), max(yr_te.max(), yhat_r.max())
    plt.plot([mn,mx],[mn,mx], linestyle="--")
    plt.xlabel("Actual total points"); plt.ylabel("Predicted total points")
    plt.title("Actual vs Predicted — Total Points")
    plt.tight_layout(); plt.savefig(REG_PNG); plt.close()

    # Feature importance (from underlying GBM)
    try:
        importances = clf_pipe.named_steps["mdl"].feature_importances_
        imp = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
        plt.figure(figsize=(7,5))
        plt.barh(imp["feature"], imp["importance"])
        plt.gca().invert_yaxis()
        plt.title("Feature Importance (Classifier)")
        plt.tight_layout(); plt.savefig(IMP_PNG); plt.close()
    except Exception:
        pass

    # Save models & meta
    joblib.dump(clf_pipe, CLF_MODEL)
    joblib.dump(reg_pipe, REG_MODEL)

    meta = {
        "clf": {"accuracy": float(acc), "roc_auc": float(auc)},
        "reg": {"mae": float(mae), "rmse": float(rmse)},
        "features": feat_names
    }
    META_FILE.write_text(json.dumps(meta, indent=2))

    print(f"[CLF] ACC={acc:.3f} AUC={auc:.3f}")
    print(f"[REG] MAE={mae:.2f} RMSE={rmse:.2f}")
    print("Saved:", CLF_MODEL.name, REG_MODEL.name, META_FILE.name, ROC_PNG.name, REG_PNG.name)

if __name__ == "__main__":
    main()