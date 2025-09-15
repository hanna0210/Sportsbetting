# app/predict.py
import joblib, xgboost as xgb
import pandas as pd
from core.features import compute_features
from core.db import load_matches
from core.config import ARTIFACT_DIR
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from core.calibration import TemperatureScaler
import numpy as np

load_dotenv()  # load DB_URL from .env

DB_URL = os.getenv("DB_URL")
engine = create_engine(DB_URL, pool_pre_ping=True)

def decimal_to_american(decimal_odds: float) -> int:
    if decimal_odds is None or decimal_odds == 0:
        return None
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1) * 100))
    else:
        return int(round(-100 / (decimal_odds - 1)))

def bookmaker_implied_probs(odd_1, odd_X, odd_2):
    def inv(x):
        try:
            return 1.0/float(x) if x and float(x) > 0 else 0.0
        except:
            return 0.0
    p1 = inv(odd_1); pX = inv(odd_X); p2 = inv(odd_2)
    s = p1 + pX + p2
    if s <= 0:
        return [1/3,1/3,1/3]
    return [p1/s, pX/s, p2/s]

def load_artifacts():
    model = joblib.load(f"{ARTIFACT_DIR}/xgb_model.joblib")
    le_country = joblib.load(f"{ARTIFACT_DIR}/le_country.joblib")
    le_league = joblib.load(f"{ARTIFACT_DIR}/le_league.joblib")
    used_features = joblib.load(f"{ARTIFACT_DIR}/features.joblib")
    temp_path = f"{ARTIFACT_DIR}/temp_scaler.joblib"
    temp_scaler = None
    if os.path.exists(temp_path):
        temp_scaler = TemperatureScaler.load(temp_path)
    return model, used_features, le_country, le_league, temp_scaler

def prepare_single_match_features(home, away, league, country, match_date, n_last=5):
    """
    Compute features for a single upcoming match by querying historical DB rows and building same features as training.
    """

    match_date_ts = pd.to_datetime(match_date)

    sql = text("""
        SELECT season, date, time, home_team, away_team, result, odd_1, "odd_X", odd_2, bets, country, league
        FROM bettingschema.odds
        WHERE date <= :dt
        ORDER BY date ASC
    """)
    with engine.connect() as conn:
        hist = pd.read_sql(sql, conn, params={'dt': match_date_ts})

    hist['date'] = pd.to_datetime(hist['date'], errors='coerce')

    # try to find exact match row (maybe bookmaker odds exist)
    match_odds = hist[
        (hist['home_team'] == home) &
        (hist['away_team'] == away) &
        (hist['league'] == league) &
        (hist['country'] == country) &
        (hist['date'] == match_date_ts)
    ].tail(1)

    if not match_odds.empty:
        odd_1, odd_X, odd_2 = match_odds.iloc[0][['odd_1','odd_X','odd_2']]
    else:
        odd_1 = odd_X = odd_2 = None

    # append row for prediction (ensures last row is our match)
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

    # compute features
    df_feats, feat_cols = compute_features(hist, n_last=n_last)
    match_row = df_feats.iloc[-1].copy()

    # load artifacts
    model, used_features, le_country, le_league, temp_scaler = load_artifacts()

    # encode categorical safely
    if str(match_row['country']) in le_country.classes_:
        match_row['country_enc'] = le_country.transform([str(match_row['country'])])[0]
    else:
        match_row['country_enc'] = -1

    if str(match_row['league']) in le_league.classes_:
        match_row['league_enc'] = le_league.transform([str(match_row['league'])])[0]
    else:
        match_row['league_enc'] = -1

    X = match_row[used_features].fillna(0).values.reshape(1, -1)
    dmat = xgb.DMatrix(X, feature_names=used_features)

    # get logits and apply temperature scaler if present
    logits = model.predict(dmat, output_margin=True)[0]  # shape (n_classes,)
    if temp_scaler is not None:
        proba = temp_scaler.transform_proba(logits.reshape(1, -1))[0]
    else:
        proba = model.predict(dmat)[0]

    # convert to fair decimal/american
    fair_decimal = [1.0/p if p>0 else None for p in proba]
    fair_american = [decimal_to_american(d) if d else None for d in fair_decimal]

    # Also provide bookmaker implied probs (normalized) and bookmaker odds (raw)
    bm_implied = bookmaker_implied_probs(odd_1, odd_X, odd_2)

    return {
        'p_home': float(proba[0]),
        'p_draw': float(proba[1]),
        'p_away': float(proba[2]),
        'odd_home_dec_fair': float(fair_decimal[0]) if fair_decimal[0] else None,
        'odd_draw_dec_fair': float(fair_decimal[1]) if fair_decimal[1] else None,
        'odd_away_dec_fair': float(fair_decimal[2]) if fair_decimal[2] else None,
        'odd_home_american': fair_american[0],
        'odd_draw_american': fair_american[1],
        'odd_away_american': fair_american[2],
        'odd_home_bookmaker': float(odd_1) if odd_1 else None,
        'odd_draw_bookmaker': float(odd_X) if odd_X else None,
        'odd_away_bookmaker': float(odd_2) if odd_2 else None,
        'bm_implied_home': float(bm_implied[0]),
        'bm_implied_draw': float(bm_implied[1]),
        'bm_implied_away': float(bm_implied[2])
    }

def prepare_upcoming_matches(n_last=5, limit=100):
    """
    Compute probabilities + fair odds for all upcoming matches (date >= today).
    """
    today = pd.Timestamp.today().normalize()

    sql = text("""
        SELECT season, date, time, home_team, away_team, result, odd_1, "odd_X", odd_2, bets, country, league
        FROM bettingschema.odds
        ORDER BY date ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    past = df[df['date'] < today].sort_values('date')
    upcoming = df[df['date'] >= today].sort_values('date').head(limit)

    results = []
    # load artifacts once
    model, used_features, le_country, le_league, temp_scaler = load_artifacts()

    for _, row in upcoming.iterrows():
        # build history = past + this upcoming row
        hist = pd.concat([past, pd.DataFrame([row])], ignore_index=True)
        hist = hist.sort_values('date').reset_index(drop=True)
        df_feats, feat_cols = compute_features(hist, n_last=n_last)
        match_row = df_feats.iloc[-1].copy()

        # encode
        if str(match_row['country']) in le_country.classes_:
            match_row['country_enc'] = le_country.transform([str(match_row['country'])])[0]
        else:
            match_row['country_enc'] = -1

        if str(match_row['league']) in le_league.classes_:
            match_row['league_enc'] = le_league.transform([str(match_row['league'])])[0]
        else:
            match_row['league_enc'] = -1

        X = match_row[used_features].fillna(0).values.reshape(1, -1)
        dmat = xgb.DMatrix(X, feature_names=used_features)

        logits = model.predict(dmat, output_margin=True)[0]
        if temp_scaler is not None:
            proba = temp_scaler.transform_proba(logits.reshape(1, -1))[0]
        else:
            proba = model.predict(dmat)[0]

        fair_decimal = [1.0/p if p>0 else None for p in proba]
        fair_american = [decimal_to_american(d) if d else None for d in fair_decimal]

        bm_probs = bookmaker_implied_probs(row.get('odd_1'), row.get('odd_X'), row.get('odd_2'))

        results.append({
            "date": str(row["date"].date()),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "league": row["league"],
            "country": row["country"],
            "p_home": float(proba[0]),
            "p_draw": float(proba[1]),
            "p_away": float(proba[2]),
            "odd_home_american": fair_american[0],
            "odd_draw_american": fair_american[1],
            "odd_away_american": fair_american[2],
            "odd_home_bookmaker": float(row.get("odd_1")) if row.get("odd_1") else None,
            "odd_draw_bookmaker": float(row.get("odd_X")) if row.get("odd_X") else None,
            "odd_away_bookmaker": float(row.get("odd_2")) if row.get("odd_2") else None,
            "bm_implied_home": float(bm_probs[0]),
            "bm_implied_draw": float(bm_probs[1]),
            "bm_implied_away": float(bm_probs[2])
        })

    return results
