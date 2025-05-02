import os 
import pandas as pd
import joblib
from datetime import datetime, timedelta

# === Load model & scaler ===
model = joblib.load("rf_model_over_4_5.joblib")
scaler = joblib.load("scaler_over_4_5.joblib")

# === Load game data ===
games = pd.read_csv("data/mlb_boxscores_full.csv")
games.columns = games.columns.str.strip().str.replace(" ", "_")
games["Game_Date"] = pd.to_datetime(games["Game_Date"])

# === Map full team names to 3-letter codes
TEAM_NAME_MAP = {
    "Athletics": "OAK", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDP", "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TBR", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSN"
}
games["Home_Team"] = games["Home_Team"].map(TEAM_NAME_MAP)
games["Away_Team"] = games["Away_Team"].map(TEAM_NAME_MAP)

# === Identify pending
innings_cols = [col for col in games.columns if any(x in col for x in ["1th", "2th", "3th", "4th", "5th"])]
games["is_pending"] = games[innings_cols].apply(lambda row: row.astype(str).str.contains("Pending", case=False).any(), axis=1)

# === Completed games only
games["Runs_1_5"] = games[innings_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
played_games = games[~games["is_pending"]].copy()
played_games["Actual_Over_4_5"] = (played_games["Runs_1_5"] > 4.5).astype(int)

# === Build 7-game rolling averages
home = played_games[["Game_Date", "Home_Team", "Runs_1_5"]].rename(columns={"Home_Team": "Team"})
away = played_games[["Game_Date", "Away_Team", "Runs_1_5"]].rename(columns={"Away_Team": "Team"})
history = pd.concat([home, away]).dropna().sort_values(["Team", "Game_Date"])
history["Form_7"] = history.groupby("Team")["Runs_1_5"].transform(lambda x: x.shift().rolling(7, min_periods=1).mean())

# === Run predictions
rows = []

for _, row in played_games.iterrows():
    game_date = row["Game_Date"].date()

    # Try multiple prior dates to locate stats
    archive_found = False
    for offset in range(1, 4):
        prior_date = (game_date - timedelta(days=offset)).strftime("%Y-%m-%d")
        std_path = f"downloads/archive/{prior_date}/team_standard.csv"
        adv_path = f"downloads/archive/{prior_date}/team_advanced.csv"

        if os.path.exists(std_path) and os.path.exists(adv_path):
            archive_found = True
            break

    if not archive_found:
        continue

    try:
        standard = pd.read_csv(std_path)
        advanced = pd.read_csv(adv_path)
    except:
        continue

    stats = pd.merge(standard, advanced, on="Tm", suffixes=("_std", "_adv")).rename(columns={"Tm": "Team"})
    home_team = row["Home_Team"]
    away_team = row["Away_Team"]

    try:
        home_stats = stats[stats["Team"] == home_team].iloc[0]
        away_stats = stats[stats["Team"] == away_team].iloc[0]
    except IndexError:
        continue

    # Recent form
    home_form = history[(history["Team"] == home_team) & (history["Game_Date"] < row["Game_Date"])]["Form_7"].max()
    away_form = history[(history["Team"] == away_team) & (history["Game_Date"] < row["Game_Date"])]["Form_7"].max()

    def extract_features(team_row):
        return [v for k, v in team_row.items() if any(stat in k for stat in [
            "BB%", "K%", "ISO", "wRC+", "OBP", "SLG", "RBI", "AVG", "OPS"
        ])]

    features = extract_features(home_stats) + extract_features(away_stats)
    features += [home_form or 0, away_form or 0]

    try:
        X_scaled = scaler.transform([features])
    except Exception:
        continue

    pred = model.predict(X_scaled)[0]
    conf = model.predict_proba(X_scaled)[0][pred]
    total = model.predict_proba(X_scaled)[0][1] * 6

    rows.append({
        "Game_Date": row["Game_Date"],
        "Home_Team": home_team,
        "Away_Team": away_team,
        "Predicted_Over_4_5": pred,
        "Confidence": round(conf, 4),
        "Model_Total": round(total, 2),
        "Actual_Over_4_5": row["Actual_Over_4_5"],
        "Runs_1_5": round(row["Runs_1_5"], 1)
    })

# === Save results
df = pd.DataFrame(rows)
df.to_csv("mlb_backfilled_predictions.csv", index=False)

if not df.empty:
    acc = (df["Predicted_Over_4_5"] == df["Actual_Over_4_5"]).mean()
    print(f"✅ Saved {len(df)} predictions to mlb_backfilled_predictions.csv")
    print(f"🎯 Accuracy: {acc:.2%}")
else:
    print("⚠️ No predictions made (missing archive data?)")
