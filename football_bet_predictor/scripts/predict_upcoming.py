import pandas as pd
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.predict import prepare_single_match_features, engine, text


# 1️⃣ Load all upcoming matches from DB
today = pd.to_datetime(datetime.today().date())

sql = text("""
    SELECT season, date, time, home_team, away_team, result, odd_1, "odd_X", odd_2, bets, country, league
    FROM bettingschema.odds
    WHERE date >= :today
    ORDER BY date ASC
""")

with engine.connect() as conn:
    upcoming_matches = pd.read_sql(sql, conn, params={'today': today})

# ensure 'date' column is datetime
upcoming_matches['date'] = pd.to_datetime(upcoming_matches['date'], errors='coerce')

# 2️⃣ Predict each match
results = []

for idx, row in upcoming_matches.iterrows():
    try:
        probs = prepare_single_match_features(
            home=row['home_team'],
            away=row['away_team'],
            league=row['league'],
            country=row['country'],
            match_date=row['date']
        )
        results.append({
            "date": row['date'],
            "home": row['home_team'],
            "away": row['away_team'],
            "p_home": probs['p_home'],
            "p_draw": probs['p_draw'],
            "p_away": probs['p_away']
        })
        print(f"Predicted {row['home_team']} vs {row['away_team']} on {row['date'].date()}")
    except Exception as e:
        print(f"Error predicting {row['home_team']} vs {row['away_team']}: {e}")

# 3️⃣ Save to CSV
df_results = pd.DataFrame(results)
df_results.to_csv("predictions.csv", index=False)
print("Saved all predictions to predictions.csv")
