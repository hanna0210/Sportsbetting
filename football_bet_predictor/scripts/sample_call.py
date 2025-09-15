# scripts/sample_call.py
import requests

# Make sure your API is running (uvicorn app.api:app --reload)
url = "http://127.0.0.1:8000/predict"
url_upcoming = "http://127.0.0.1:8000/predict_upcoming"

# Example: Get all upcoming matches
resp = requests.get(url_upcoming, params={"limit": 20})
data = resp.json()

if not data.get("success", False):
    print("API error:", data.get("error"))
else:
    print("------ Upcoming Matches ------")
    for match in data["matches"]:
        print(f"{match['date']} | {match['home_team']} vs {match['away_team']} ({match['league']})")
        print(f"  Probabilities: H={match['p_home']:.3f}, D={match['p_draw']:.3f}, A={match['p_away']:.3f}")
        print(f"  Fair Odds (American): H={match['odd_home_american']}, D={match['odd_draw_american']}, A={match['odd_away_american']}")
        print(f"  Bookmaker Odds: H={match['odd_home_bookmaker']}, D={match['odd_draw_bookmaker']}, A={match['odd_away_bookmaker']}")
        print(f"  Bookmaker implied probs: H={match['bm_implied_home']:.3f}, D={match['bm_implied_draw']:.3f}, A={match['bm_implied_away']:.3f}")
        print("-"*60)



# ------ Upcoming Matches ------
# 2025-09-14 | Fluminense vs Corinthians (Serie A Betano)
#   Probabilities: H=0.348, D=0.311, A=0.341
#   Fair Odds (American): H=187, D=221, A=193
#   Bookmaker Odds: H=102.0, D=206.0, A=313.0
#   Bookmaker implied probs: H=0.549, D=0.272, A=0.179
# ------------------------------------------------------------
# 2025-09-14 | Heracles vs AZ Alkmaar (Eredivisie)
#   Probabilities: H=0.208, D=0.209, A=0.583
#   Fair Odds (American): H=381, D=379, A=-140
#   Bookmaker Odds: H=463.0, D=338.0, A=-189.0
#   Bookmaker implied probs: H=0.422, D=0.578, A=0.000
# ------------------------------------------------------------
# 2025-09-14 | Utrecht vs Groningen (Eredivisie)
#   Probabilities: H=0.623, D=0.205, A=0.172
#   Fair Odds (American): H=-165, D=388, A=481
#   Bookmaker Odds: H=-137.0, D=298.0, A=342.0
#   Bookmaker implied probs: H=0.000, D=0.534, A=0.466
# ------------------------------------------------------------
# 2025-09-14 | Telstar vs Sittard (Eredivisie)
#   Probabilities: H=0.390, D=0.275, A=0.335
#   Fair Odds (American): H=156, D=264, A=199
#   Bookmaker Odds: H=139.0, D=244.0, A=188.0
#   Bookmaker implied probs: H=0.433, D=0.247, A=0.320
# ------------------------------------------------------------
# 2025-09-14 | Antwerp vs Gent (Jupiler Pro League)
#   Probabilities: H=0.416, D=0.271, A=0.313
#   Fair Odds (American): H=140, D=268, A=220
#   Bookmaker Odds: H=131.0, D=245.0, A=189.0
#   Bookmaker implied probs: H=0.449, D=0.240, A=0.311
# ------------------------------------------------------------
# 2025-09-14 | Cercle Brugge KSV vs Charleroi (Jupiler Pro League)
#   Probabilities: H=0.429, D=0.259, A=0.311
#   Fair Odds (American): H=133, D=285, A=221
#   Bookmaker Odds: H=121.0, D=250.0, A=199.0
#   Bookmaker implied probs: H=0.478, D=0.231, A=0.291
# ------------------------------------------------------------
# 2025-09-14 | Anderlecht vs Genk (Jupiler Pro League)
#   Probabilities: H=0.401, D=0.271, A=0.328
#   Fair Odds (American): H=150, D=269, A=205
#   Bookmaker Odds: H=141.0, D=252.0, A=172.0
#   Bookmaker implied probs: H=0.420, D=0.235, A=0.345
# ------------------------------------------------------------
# 2025-09-14 | St. Truiden vs Westerlo (Jupiler Pro League)
#   Probabilities: H=0.364, D=0.386, A=0.250
#   Fair Odds (American): H=175, D=159, A=300
#   Bookmaker Odds: H=106.0, D=272.0, A=215.0
#   Bookmaker implied probs: H=0.531, D=0.207, A=0.262
# ------------------------------------------------------------
# 2025-09-14 | Kayserispor vs Goztepe (Super Lig)
#   Probabilities: H=0.284, D=0.325, A=0.391
#   Fair Odds (American): H=252, D=207, A=156
#   Bookmaker Odds: H=224.0, D=216.0, A=127.0
#   Bookmaker implied probs: H=0.263, D=0.273, A=0.464
# ------------------------------------------------------------
# 2025-09-14 | Fenerbahce vs Trabzonspor (Super Lig)
#   Probabilities: H=0.746, D=0.176, A=0.078
#   Fair Odds (American): H=-294, D=469, A=1182
#   Bookmaker Odds: H=-238.0, D=367.0, A=554.0
#   Bookmaker implied probs: H=0.000, D=0.602, A=0.398
# ------------------------------------------------------------
# 2025-09-14 | Gaziantep vs Kocaelispor (Super Lig)
#   Probabilities: H=0.501, D=0.265, A=0.234
#   Fair Odds (American): H=-100, D=278, A=327
#   Bookmaker Odds: H=114.0, D=227.0, A=246.0
#   Bookmaker implied probs: H=0.509, D=0.255, A=0.236
# ------------------------------------------------------------
# 2025-09-14 | Krylya Sovetov vs Sochi (Premier League)
#   Probabilities: H=0.423, D=0.296, A=0.281
#   Fair Odds (American): H=136, D=238, A=256
#   Bookmaker Odds: H=110.0, D=250.0, A=229.0
#   Bookmaker implied probs: H=0.521, D=0.229, A=0.250
# ------------------------------------------------------------
# 2025-09-14 | Excelsior vs Sparta Rotterdam (Eredivisie)
#   Probabilities: H=0.334, D=0.268, A=0.398
#   Fair Odds (American): H=199, D=274, A=151
#   Bookmaker Odds: H=192.0, D=252.0, A=131.0
#   Bookmaker implied probs: H=0.310, D=0.236, A=0.454
# ------------------------------------------------------------
# 2025-09-14 | Baltika vs Zenit (Premier League)
#   Probabilities: H=0.136, D=0.257, A=0.607
#   Fair Odds (American): H=637, D=289, A=-155
#   Bookmaker Odds: H=400.0, D=295.0, A=-161.0
#   Bookmaker implied probs: H=0.424, D=0.576, A=0.000
# ------------------------------------------------------------
# 2025-09-14 | Kryvbas vs Polissya Zhytomyr (Premier League)
#   Probabilities: H=0.418, D=0.268, A=0.314
#   Fair Odds (American): H=139, D=273, A=218
#   Bookmaker Odds: H=125.0, D=204.0, A=218.0
#   Bookmaker implied probs: H=0.457, D=0.280, A=0.262
# ------------------------------------------------------------
# 2025-09-14 | Karpaty Lviv vs SC Poltava (Premier League)
#   Probabilities: H=0.593, D=0.259, A=0.148
#   Fair Odds (American): H=-146, D=286, A=574
#   Bookmaker Odds: H=-182.0, D=295.0, A=426.0
#   Bookmaker implied probs: H=0.000, D=0.591, A=0.409
# ------------------------------------------------------------
# 2025-09-14 | Motor Lublin vs Termalica B-B. (Ekstraklasa)
#   Probabilities: H=0.468, D=0.238, A=0.294
#   Fair Odds (American): H=114, D=320, A=240
#   Bookmaker Odds: H=110.0, D=252.0, A=214.0
#   Bookmaker implied probs: H=0.513, D=0.224, A=0.264
# ------------------------------------------------------------
# 2025-09-14 | Legia vs Radomiak Radom (Ekstraklasa)
#   Probabilities: H=0.629, D=0.223, A=0.148
#   Fair Odds (American): H=-169, D=348, A=577
#   Bookmaker Odds: H=-196.0, D=308.0, A=461.0
#   Bookmaker implied probs: H=0.000, D=0.599, A=0.401
# ------------------------------------------------------------
# 2025-09-14 | Widzew Lodz vs Arka Gdynia (Ekstraklasa)
#   Probabilities: H=0.557, D=0.253, A=0.190
#   Fair Odds (American): H=-126, D=296, A=425
#   Bookmaker Odds: H=-135.0, D=272.0, A=328.0
#   Bookmaker implied probs: H=0.000, D=0.547, A=0.453