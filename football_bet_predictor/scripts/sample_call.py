# scripts/sample_call.py
import requests

# Make sure your API is running (uvicorn app.api:app --reload)
url = "http://127.0.0.1:8000/predict"
url_upcoming = "http://127.0.0.1:8000/predict_upcoming"
# payload = {
#     "home_team": "Brighton",
#     "away_team": "Tottenham",
#     "league": "Premier League",
#     "country": "England",
#     "date": "2025-09-20"   # format YYYY-MM-DD
# }

# response = requests.post(url, json=payload)
# data = response.json()

# print("Raw response:", data)

# if not data.get("success", False):
#     print("API error:", data.get("error"))
# else:
#     print("------ Prediction ------")
#     print(f"Home win probability: {data['p_home']:.3f}")
#     print(f"Draw probability:     {data['p_draw']:.3f}")
#     print(f"Away win probability: {data['p_away']:.3f}")

#     print("\n------ Fair Odds (American) ------")
#     print(f"Home win odds: {data['odd_home_american']}")
#     print(f"Draw odds:     {data['odd_draw_american']}")
#     print(f"Away win odds: {data['odd_away_american']}")


# 🔹 Example: Get all upcoming matches
resp = requests.get(url_upcoming, params={"limit": 10})
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
        print("-"*60)