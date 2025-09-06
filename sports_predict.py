# sports_predict.py — Predict win probability (home) & total points for upcoming fixtures
# Inputs: fixtures.csv, sports_clf.pkl, sports_reg.pkl, sports_meta.json
# Output: prediction.csv  (win_prob_home, win_prob_away, pred_total_points, optional O/U suggestion)

from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import joblib

BASE = Path(".").resolve()
FIXTURES = BASE/"fixtures.csv"
CLF_MODEL = BASE/"sports_clf.pkl"
REG_MODEL = BASE/"sports_reg.pkl"
META_FILE = BASE/"sports_meta.json"
OUT_FILE = BASE/"prediction.csv"

def load_csv_any(p: Path) -> pd.DataFrame:
    for sep in (None, ",", ";", "\t"):
        try:
            return pd.read_csv(p, sep=sep, engine="python")
        except Exception:
            pass
    raise FileNotFoundError(f"Cannot read {p}")

def build_features_from_fixtures(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    df = df.copy()

    # Feature diffs (home - away) — πρέπει να ταιριάζουν με το training
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

    # Κράτα ΜΟΝΟ τα features που περίμενε το μοντέλο
    X = df.reindex(columns=feature_names, fill_value=np.nan)
    return X

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--line", type=float, default=None,
                        help=" προαιρετική γραμμή Over/Under (π.χ. --line 160.5)")
    args = parser.parse_args()

    # Φόρτωση μεταδεδομένων & μοντέλων
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    feat_names = meta["features"]

    clf = joblib.load(CLF_MODEL)
    reg = joblib.load(REG_MODEL)

    # Διαβάζουμε fixtures
    df_fix = load_csv_any(FIXTURES)

    # Φτιάχνουμε X (ίδια features με training)
    X = build_features_from_fixtures(df_fix, feat_names)

    # Προβλέψεις
    win_prob_home = clf.predict_proba(X)[:, 1]
    win_prob_away = 1.0 - win_prob_home
    pred_total = reg.predict(X)

    # Συγκεντρώνουμε αποτελέσματα
    out = df_fix[["date","league","home_team","away_team"]].copy()
    out["win_prob_home"] = (win_prob_home * 100).round(1)
    out["win_prob_away"] = (win_prob_away * 100).round(1)
    out["pred_total_points"] = pred_total.round(1)

    # Προαιρετικές προτάσεις
    suggestions = []
    for p_home, total in zip(win_prob_home, pred_total):
        pick = "Home ML" if p_home >= 0.55 else ("Away ML" if p_home <= 0.45 else "No bet (side)")
        if args.line is not None:
            if total >= args.line + 1.5:
                ou = f"Over {args.line}"
            elif total <= args.line - 1.5:
                ou = f"Under {args.line}"
            else:
                ou = "No bet (O/U)"
            pick = f"{pick} · {ou}"
        suggestions.append(pick)
    out["suggestion"] = suggestions

    out.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"Saved -> {OUT_FILE}")

if __name__ == "__main__":
    main()