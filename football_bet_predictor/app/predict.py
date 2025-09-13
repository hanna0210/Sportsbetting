import joblib, xgboost as xgb
import pandas as pd
from core.features import compute_features
from core.db import load_matches
from core.config import ARTIFACT_DIR
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()  # load DB_URL from .env

DB_URL = os.getenv("DB_URL")
engine = create_engine(DB_URL, pool_pre_ping=True)

def load_artifacts():
    model = joblib.load(f"{ARTIFACT_DIR}/xgb_model.joblib")
    le_country = joblib.load(f"{ARTIFACT_DIR}/le_country.joblib")
    le_league = joblib.load(f"{ARTIFACT_DIR}/le_league.joblib")
    used_features = joblib.load(f"{ARTIFACT_DIR}/features.joblib")
    return model, used_features, le_country, le_league

def decimal_to_american(decimal_odds: float) -> int:
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))
    
def prepare_single_match_features(home, away, league, country, match_date, n_last=5):
    match_date_ts = pd.to_datetime(match_date)

    # 1️⃣ Load historical matches
    sql = text("""
        SELECT season, date, time, home_team, away_team, result,
               odd_1, "odd_X", odd_2, bets, country, league
        FROM bettingschema.odds
        WHERE date <= :dt
        ORDER BY date ASC
    """)
    with engine.connect() as conn:
        hist = pd.read_sql(sql, conn, params={'dt': match_date_ts})

    hist['date'] = pd.to_datetime(hist['date'], errors='coerce')

    # 2️⃣ Try to locate existing odds for this match (if already in DB)
    match_odds = hist[
        (hist['home_team'] == home) &
        (hist['away_team'] == away) &
        (hist['league'] == league) &
        (hist['country'] == country) &
        (hist['date'] == match_date_ts)
    ].tail(1)

    if not match_odds.empty:
        odd_1, odd_X, odd_2 = match_odds[['odd_1', 'odd_X', 'odd_2']].values[0]
    else:
        odd_1 = odd_X = odd_2 = None

    # 3️⃣ Add dummy row if not in DB
    if match_odds.empty:
        new_row = {
            'season': None,
            'date': match_date_ts,
            'time': None,
            'home_team': home,
            'away_team': away,
            'result': None,
            'odd_1': odd_1,
            'odd_X': odd_X,
            'odd_2': odd_2,
            'bets': None,
            'country': country,
            'league': league
        }
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        hist = hist.sort_values('date').reset_index(drop=True)

    # 4️⃣ Compute features
    df_feats, feat_cols = compute_features(hist, n_last=n_last)
    match_row = df_feats.iloc[-1].copy()

    # 5️⃣ Load model + encoders
    model, used_features, le_country, le_league = load_artifacts()

    # Encode country/league
    match_row['country_enc'] = le_country.transform([str(match_row['country'])])[0] \
        if str(match_row['country']) in le_country.classes_ else -1
    match_row['league_enc'] = le_league.transform([str(match_row['league'])])[0] \
        if str(match_row['league']) in le_league.classes_ else -1
    
     # 6️⃣ Predict
    X = match_row[used_features].fillna(0).values.reshape(1, -1)
    dmat = xgb.DMatrix(X, feature_names=used_features)
    proba = model.predict(dmat)[0]  # [p_home, p_draw, p_away]

    # 7️⃣ Convert probs → decimal odds → American odds
    fair_decimal = [1/p if p > 0 else None for p in proba]
    fair_american = [decimal_to_american(d) if d else None for d in fair_decimal]

    return {
        'p_home': float(proba[0]),
        'p_draw': float(proba[1]),
        'p_away': float(proba[2]),
        'odd_home_american': fair_american[0],
        'odd_draw_american': fair_american[1],
        'odd_away_american': fair_american[2]
    }
