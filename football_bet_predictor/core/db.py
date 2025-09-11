import pandas as pd
from sqlalchemy import create_engine, text
from .config import DB_URL

engine = create_engine(DB_URL, pool_pre_ping=True)

def load_matches(up_to_date=None):
    sql = """
    SELECT season, date, time, home_team, away_team, result, odd_1, "odd_X", odd_2, bets, country, league
    FROM bettingschema.odds
    WHERE date IS NOT NULL
    ORDER BY date ASC;
    """
    df = pd.read_sql(text(sql), engine) 
    df['date'] = pd.to_datetime(df['date'])
    if up_to_date:
        df = df[df['date'] < pd.to_datetime(up_to_date)]
    return df
