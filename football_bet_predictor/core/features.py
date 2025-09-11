import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --------- ELO functions ---------
def init_elo(df, start_rating=1500):
    teams = pd.unique(df[['home_team','away_team']].values.ravel())
    return {t: start_rating for t in teams}

def expected_score(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

def update_elo(elo, home, away, score, k_home=20, k_away=20, home_field_advantage=100):
    Ra = elo.get(home,1500) + home_field_advantage
    Rb = elo.get(away,1500)
    Ea = expected_score(Ra, Rb)
    Sb = 1 - score
    elo[home] = elo.get(home,1500) + k_home * (score - Ea)
    elo[away] = elo.get(away,1500) + k_away * (Sb - (1-Ea))
    return elo

# --------- Feature engineering ---------
def parse_result_to_label(res):
    if pd.isna(res): return np.nan
    s = str(res).strip()
    if s.lower() in ('h','home'): return 0
    if s.lower() in ('d','draw'): return 1
    if s.lower() in ('a','away'): return 2
    if '-' in s or ':' in s:
        sep = '-' if '-' in s else ':'
        try:
            a,b = map(int, s.split(sep))
            return 0 if a>b else 1 if a==b else 2
        except:
            return np.nan
    return np.nan

def compute_features(df, n_last=5):
    """
    Return features + final columns.
    """
    df = df.copy().reset_index(drop=True)
    df['target'] = df['result'].apply(parse_result_to_label)

    # Implied probabilities
    def implied_prob(row):
        try:
            p1 = 1/row['odd_1'] if row['odd_1']>0 else np.nan
            pX = 1/row['odd_X'] if row['odd_X']>0 else np.nan
            p2 = 1/row['odd_2'] if row['odd_2']>0 else np.nan
            s = sum([x for x in (p1,pX,p2) if pd.notna(x)])
            return pd.Series([p1/s if pd.notna(p1) else 0,
                              pX/s if pd.notna(pX) else 0,
                              p2/s if pd.notna(p2) else 0])
        except: return pd.Series([np.nan,np.nan,np.nan])

    df[['imp_prob_H','imp_prob_D','imp_prob_A']] = df.apply(implied_prob, axis=1)

    # ELO & rolling stats
    elo = init_elo(df)
    home_elos=[]; away_elos=[]
    history={}

    for idx,row in df.iterrows():
        h,a = row['home_team'], row['away_team']
        home_elos.append(elo.get(h,1500))
        away_elos.append(elo.get(a,1500))

        # Recent form
        def last_stats(team):
            last = history.get(team, [])[-n_last:]
            if not last: return {'form_wins':0,'form_points':0,'form_matches':0}
            pts = sum([3 if r==1 else 1 if r==0.5 else 0 for r in last])
            wins = sum([1 for r in last if r==1])
            return {'form_wins': wins/len(last), 'form_points': pts/(3*len(last)), 'form_matches': len(last)}

        hstats = last_stats(h)
        astats = last_stats(a)
        for k,v in hstats.items(): df.at[idx,f'h_{k}']=v
        for k,v in astats.items(): df.at[idx,f'a_{k}']=v

        # H2H advantage
        mask = ((df.loc[:idx-1,'home_team']==h) & (df.loc[:idx-1,'away_team']==a)) | ((df.loc[:idx-1,'home_team']==a) & (df.loc[:idx-1,'away_team']==h))
        last_h2h = df.loc[:idx-1].loc[mask].tail(n_last)
        wins = 0
        for _,r in last_h2h.iterrows():
            lab = parse_result_to_label(r['result'])
            if lab==0 and r['home_team']==h: wins+=1
            if lab==2 and r['away_team']==h: wins+=1
            if lab==0 and r['home_team']==a: wins-=1
            if lab==2 and r['away_team']==a: wins-=1
        df.at[idx,'h2h_adv'] = wins/len(last_h2h) if len(last_h2h)>0 else 0

        # update ELO/history
        t = df.at[idx,'target']
        if pd.notna(t):
            score = 1 if t==0 else 0.5 if t==1 else 0
            update_elo(elo,h,a,score)
            history.setdefault(h,[]).append(score)
            history.setdefault(a,[]).append(1-score)

    df['elo_home']=home_elos
    df['elo_away']=away_elos
    df['elo_diff']=df['elo_home']-df['elo_away']

    # Fill NAs
    for col in ['imp_prob_H','imp_prob_D','imp_prob_A','h_form_wins','h_form_points','a_form_wins','a_form_points','h2h_adv']:
        df[col] = df[col].fillna(0.33 if 'prob' in col else 0.0)

    feat_cols = ['elo_home','elo_away','elo_diff','h_form_wins','h_form_points','a_form_wins','a_form_points','h2h_adv','imp_prob_H','imp_prob_D','imp_prob_A','country','league']
    return df, feat_cols
