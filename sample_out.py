import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "home_team": "Manchester United",
    "away_team": "Liverpool",
    "league": "Premier League",
    "country": "England",
    "date": "2025-09-20"
}

response = requests.post(url, json=payload)
data = response.json()

print("Raw response:", data)

if not data["success"]:
    print("API error:", data["error"])
else:
    print("Home win probability:", data["p_home"])
    print("Draw probability:", data["p_draw"])
    print("Away win probability:", data["p_away"])
