# make_report.py — Sports PDF report (classification + regression + predictions)
# Inputs: sports_meta.json, prediction.csv, roc_curve.png, reg_scatter.png, feature_importance.png
# Output: sports_report.pdf

from pathlib import Path
import json
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

BASE = Path(".").resolve()
META_FILE = BASE/"sports_meta.json"
PRED_FILE = BASE/"prediction.csv"
PDF_FILE  = BASE/"sports_report.pdf"

ROC_IMG = BASE/"roc_curve.png"
REG_IMG = BASE/"reg_scatter.png"
IMP_IMG = BASE/"feature_importance.png"

def safe_image(path: Path, width=460):
    if path.exists():
        img = Image(str(path))
        img._restrictSize(width, 9999)  # keep aspect ratio
        return img
    return Paragraph(f"<i>Image not found: {path.name}</i>", getSampleStyleSheet()["Italic"])

def fmt_pct(x):
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return str(x)

def build_predictions_table(df: pd.DataFrame, max_rows=10):
    # keep only relevant cols if present
    cols_pref = ["date","league","home_team","away_team","win_prob_home","win_prob_away","pred_total_points","suggestion"]
    cols = [c for c in cols_pref if c in df.columns]
    df2 = df[cols].copy()

    # format percentages if they look numeric
    for c in ["win_prob_home","win_prob_away"]:
        if c in df2.columns:
            df2[c] = df2[c].apply(fmt_pct)

    # limit rows
    if len(df2) > max_rows:
        df_show = df2.head(max_rows)
        note = Paragraph(f"<i>Showing first {max_rows} rows of {len(df2)} total.</i>", getSampleStyleSheet()["Italic"])
    else:
        df_show = df2
        note = Spacer(0, 0)

    data = [list(df_show.columns)] + df_show.values.tolist()
    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f0f0f0")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#000000")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ALIGN", (0,0), (-1,0), "LEFT"),
        ("ALIGN", (-3,1), (-2,-1), "RIGHT"),  # win probs
        ("ALIGN", (-3,1), (-3,-1), "RIGHT"),  # pred_total_points if column index matches
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#fbfbfb")]),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    return tbl, note

def main():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleBig", parent=styles["Title"], fontSize=20, spaceAfter=8)
    h_style = ParagraphStyle("Heading", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6)
    body = styles["BodyText"]
    italic = styles["Italic"]

    story = []
    story.append(Paragraph("Sports Match Outcome Prediction — Report", title_style))
    story.append(Paragraph("Basketball: Home Win Probability & Total Points (Over/Under)", italic))
    story.append(Spacer(0, 12))

    # Load meta (metrics)
    if META_FILE.exists():
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        clf_acc = meta.get("clf", {}).get("accuracy", None)
        clf_auc = meta.get("clf", {}).get("roc_auc", None)
        reg_mae = meta.get("reg", {}).get("mae", None)
        reg_rmse = meta.get("reg", {}).get("rmse", None)

        story.append(Paragraph("Model Performance", h_style))
        items = []
        if clf_acc is not None: items.append(f"Classification Accuracy: <b>{clf_acc:.3f}</b>")
        if clf_auc is not None and clf_auc == clf_auc: items.append(f"ROC–AUC: <b>{clf_auc:.3f}</b>")
        if reg_mae is not None: items.append(f"Regression MAE (total points): <b>{reg_mae:.2f}</b>")
        if reg_rmse is not None: items.append(f"Regression RMSE (total points): <b>{reg_rmse:.2f}</b>")
        if items:
            story.append(Paragraph(" • " + "<br/> • ".join(items), body))
        story.append(Spacer(0, 10))
    else:
        story.append(Paragraph("<i>metrics file not found (sports_meta.json)</i>", italic))

    # Images
    story.append(Paragraph("ROC Curve (Classifier)", h_style))
    story.append(safe_image(ROC_IMG, width=420))
    story.append(Spacer(0, 6))

    story.append(Paragraph("Feature Importance (Classifier)", h_style))
    story.append(safe_image(IMP_IMG, width=420))
    story.append(Spacer(0, 6))

    story.append(Paragraph("Actual vs Predicted — Total Points (Regressor)", h_style))
    story.append(safe_image(REG_IMG, width=420))
    story.append(PageBreak())

    # Predictions table
    story.append(Paragraph("Predictions (Upcoming Fixtures)", h_style))
    if PRED_FILE.exists():
        df_pred = pd.read_csv(PRED_FILE)
        tbl, note = build_predictions_table(df_pred, max_rows=15)
        story.append(tbl)
        story.append(Spacer(0, 6))
        story.append(note)
    else:
        story.append(Paragraph("<i>prediction.csv not found</i>", italic))

    # Executive summary
    story.append(Spacer(0, 14))
    story.append(Paragraph("Executive Summary", h_style))
    summary_text = (
        "The pipeline estimates <b>home win probabilities</b> and <b>total points</b> for upcoming games. "
        "Key drivers (per feature importance) often include <b>FG% differential</b>, <b>TOV differential</b>, and <b>PPG differential</b>. "
        "Use win probabilities (>55% as a soft threshold) and predicted totals versus market lines to form <i>indicative</i> suggestions."
    )
    story.append(Paragraph(summary_text, body))

    # Build PDF
    doc = SimpleDocTemplate(str(PDF_FILE), pagesize=A4, leftMargin=28, rightMargin=28, topMargin=30, bottomMargin=30)
    doc.build(story)
    print(f"Saved -> {PDF_FILE}")

if __name__ == "__main__":
    main()