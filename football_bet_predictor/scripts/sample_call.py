# scripts/sample_call.py
import requests

# Make sure your API is running (uvicorn app.api:app --reload)
url = "http://127.0.0.1:8000/predict"

payload = {
    "home_team": "Brighton",
    "away_team": "Tottenham",
    "league": "Premier League",
    "country": "England",
    "date": "2025-09-20"   # format YYYY-MM-DD
}

response = requests.post(url, json=payload)
data = response.json()

print("Raw response:", data)

if not data.get("success", False):
    print("API error:", data.get("error"))
else:
    print("------ Prediction ------")
    print(f"Home win probability: {data['p_home']:.3f}")
    print(f"Draw probability:     {data['p_draw']:.3f}")
    print(f"Away win probability: {data['p_away']:.3f}")

    print("\n------ Fair Odds (American) ------")
    print(f"Home win odds: {data['odd_home_american']}")
    print(f"Draw odds:     {data['odd_draw_american']}")
    print(f"Away win odds: {data['odd_away_american']}")
