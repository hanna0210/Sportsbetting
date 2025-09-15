# odds_multi_countries_fixed_dates.py
import time
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from decimal import Decimal
from datetime import datetime, date


import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    StaleElementReferenceException,
    ElementClickInterceptedException,
)

import psycopg2
from psycopg2.extras import execute_values

# -------------------- DB CONFIG --------------------
DB_CONFIG = {
    "host":     "ep-empty-term-adh8v9zs-pooler.c-2.us-east-1.aws.neon.tech",
    "port":     5432,
    "dbname":   "sportsbetting",
    "user":     "neondb_owner",
    "password": "npg_ZdRSbgLjv5s6",
}

TABLE  = "odds"

# -------------------- League configuration --------------------
@dataclass
class LeagueConfig:
    country: str               # url segment (brazil, england, ...)
    league_name: str           # label stored in DB
    base: str                  # e.g. https://www.oddsportal.com/football/germany/
    kind: str                  # 'single_year' | 'two_year'
    comp_slug: str             # results slug (bundesliga, serie-a, laliga, ...)
    seasons: List[int]         # single_year: [2021,...] ; two_year: [2021,...] (means 2021-22)
    next_slug: Optional[str] = None          # slug for "next matches" page
    special_slugs: Dict[int, str] = None     # overrides for certain seasons

    def results_url(self, start_year: int) -> str:
        if self.special_slugs and start_year in self.special_slugs:
            return self.base + self.special_slugs[start_year]
        if self.kind == "single_year":
            return f"{self.base}{self.comp_slug}-{start_year}/results/"
        # two-year
        if start_year >= 2025:
            return f"{self.base}{self.comp_slug}/results/"  # seasonless current page
        return f"{self.base}{self.comp_slug}-{start_year}-{start_year+1}/results/"

    def next_url(self) -> str:
        slug = self.next_slug or self.comp_slug
        return f"{self.base}{slug}/"

# --- Leagues ---
BRAZIL = LeagueConfig(
    country="brazil",
    league_name="Serie A Betano",
    base="https://www.oddsportal.com/football/brazil/",
    kind="single_year",
    comp_slug="serie-a",
    seasons=[2021, 2022, 2023, 2024, 2025],
    next_slug="serie-a-betano",
    special_slugs={2024: "serie-a-betano-2024/results/", 2025: "serie-a-betano/results/"},
)
ENGLAND = LeagueConfig("england", "Premier League", "https://www.oddsportal.com/football/england/", "two_year", "premier-league", [2021, 2022, 2023, 2024, 2025])
SPAIN   = LeagueConfig("spain",   "LaLiga",          "https://www.oddsportal.com/football/spain/",   "two_year", "laliga",          [2021, 2022, 2023, 2024, 2025])
GERMANY = LeagueConfig("germany", "Bundesliga",      "https://www.oddsportal.com/football/germany/","two_year", "bundesliga",      [2021, 2022, 2023, 2024, 2025])
ITALY   = LeagueConfig("italy",   "Serie A",         "https://www.oddsportal.com/football/italy/",  "two_year", "serie-a",         [2021, 2022, 2023, 2024, 2025])
FRANCE  = LeagueConfig("france",  "Ligue 1",         "https://www.oddsportal.com/football/france/", "two_year", "ligue-1",         [2021, 2022, 2023, 2024, 2025])
# ARGENTINA = LeagueConfig("argentina","Torneo Betano","https://www.oddsportal.com/football/argentina/","single_year","torneo-betano",[2021, 2022, 2023, 2024, 2025])
PORTUGAL  = LeagueConfig("portugal","Liga Portugal", "https://www.oddsportal.com/football/portugal/","two_year","liga-portugal",[2021, 2022, 2023, 2024, 2025])
NETHERLANDS = LeagueConfig("netherlands","Eredivisie","https://www.oddsportal.com/football/netherlands/","two_year","eredivisie",[2021, 2022, 2023, 2024, 2025])
BELGIUM  = LeagueConfig("belgium","Jupiler Pro League","https://www.oddsportal.com/football/belgium/","two_year","jupiler-pro-league",[2021, 2022, 2023, 2024, 2025])
TURKEY   = LeagueConfig("turkey","Super Lig","https://www.oddsportal.com/football/turkey/","two_year","super-lig",[2021, 2022, 2023, 2024, 2025])
RUSSIA   = LeagueConfig("russia","Premier League","https://www.oddsportal.com/football/russia/","two_year","premier-league",[2021, 2022, 2023, 2024, 2025])
UKRAINE  = LeagueConfig("ukraine","Premier League","https://www.oddsportal.com/football/ukraine/","two_year","premier-league",[2021, 2022, 2023, 2024, 2025])
POLAND   = LeagueConfig("poland","Ekstraklasa","https://www.oddsportal.com/football/poland/","two_year","ekstraklasa",[2021, 2022, 2023, 2024, 2025])
# CZECH    = LeagueConfig("czech-republic","Fortuna Liga","https://www.oddsportal.com/football/czech-republic/","two_year","fortuna-liga",[2021, 2022, 2023, 2024, 2025])
AUSTRIA  = LeagueConfig("austria","Bundesliga","https://www.oddsportal.com/football/austria/","two_year","bundesliga",[2021, 2022, 2023, 2024, 2025])
SWITZERLAND = LeagueConfig("switzerland","Super League","https://www.oddsportal.com/football/switzerland/","two_year","super-league",[2021, 2022, 2023, 2024, 2025])

LEAGUES: List[LeagueConfig] = [
    BRAZIL, ENGLAND, SPAIN,
    GERMANY, ITALY, FRANCE,
    PORTUGAL, NETHERLANDS, BELGIUM, TURKEY, 
    RUSSIA, UKRAINE, POLAND, AUSTRIA, SWITZERLAND, 
]

# -------------------- Selenium setup --------------------
def make_driver(headless: bool = True) -> uc.Chrome:
    chrome_opts = Options()
    if headless:
        chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--window-size=1600,1400")
    chrome_opts.add_argument("--disable-dev-shm-usage")
    chrome_opts.add_argument("--lang=en-US")
    chrome_opts.add_argument("--disable-blink-features=AutomationControlled")
    chrome_opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    )
    driver = uc.Chrome(version_main=139, options=chrome_opts)
    driver.set_page_load_timeout(60)
    return driver

def wait_for_results_table(driver):
    WebDriverWait(driver, 25).until(
        EC.presence_of_element_located(
            (By.XPATH, "//div[@data-testid='secondary-header'] | //div[@data-testid='game-row']")
        )
    )
    time.sleep(0.4)

def close_popups(driver):
    for by, sel in [
        (By.XPATH, "//button[contains(., 'Accept') or contains(.,'I Agree') or contains(.,'I accept')]"),
        (By.CSS_SELECTOR, "div[role='dialog'] button"),
        (By.XPATH, "//div[contains(@class,'cookie')]//button"),
    ]:
        try:
            el = WebDriverWait(driver, 2).until(EC.element_to_be_clickable((by, sel)))
            el.click()
            time.sleep(0.2)
        except Exception:
            pass

def go_to_url(driver, url: str):
    driver.get(url)
    wait_for_results_table(driver)
    close_popups(driver)

def _row_count(driver) -> int:
    return len(driver.find_elements(
        By.XPATH,
        "//div[@data-testid='game-row']/ancestor::div[contains(@class,'group') and contains(@class,'flex')]"
    ))

def scroll_to_bottom_until_stable(driver, *, expected_rows_per_page=50, min_stable_checks=2, max_loops=40, pause=0.25) -> int:
    stable = 0
    prev_h = -1
    prev_rows = -1
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(0.2)
    for _ in range(max_loops):
        driver.execute_script("window.scrollBy(0, Math.max(700, window.innerHeight*0.9));")
        time.sleep(pause)
        at_bottom = driver.execute_script("return (window.innerHeight + window.scrollY) >= (document.body.scrollHeight - 4);")
        cur_h = driver.execute_script("return document.body.scrollHeight;")
        cur_rows = _row_count(driver)
        stable = stable + 1 if (cur_h == prev_h and cur_rows == prev_rows) else 0
        prev_h, prev_rows = cur_h, cur_rows
        if at_bottom and (cur_rows >= expected_rows_per_page or stable >= min_stable_checks):
            break
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(0.2)
    return _row_count(driver)

def locate_next_button(driver):
    xp = "//a[contains(concat(' ', normalize-space(@class), ' '), ' pagination-link ') and normalize-space(.)='Next']"
    links = driver.find_elements(By.XPATH, xp)
    return links[-1] if links else None

def click_next_page(driver) -> bool:
    scroll_to_bottom_until_stable(driver)
    close_popups(driver)
    btn = locate_next_button(driver)
    if not btn:
        return False
    driver.execute_script("arguments[0].scrollIntoView({block:'center', inline:'nearest'});", btn)
    driver.execute_script("window.scrollBy(0, -140);")
    time.sleep(0.1)
    try:
        btn.click()
    except ElementClickInterceptedException:
        driver.execute_script("arguments[0].click();", btn)
    except Exception:
        driver.execute_script("arguments[0].click();", btn)
    wait_for_results_table(driver)
    return True

def get_total_pages(driver) -> Optional[int]:
    scroll_to_bottom_until_stable(driver)
    xp = ("//a[contains(concat(' ', normalize-space(@class), ' '), ' pagination-link ') "
          "and normalize-space(.)!='Next' and normalize-space(.)!='Previous']")
    nums = []
    for el in driver.find_elements(By.XPATH, xp):
        txt = (el.text or "").strip()
        if txt.isdigit():
            nums.append(int(txt))
    return max(nums) if nums else None

# -------------------- Date handling --------------------
# Accepts “24 Apr 2022”, “24 April 2022”, even when surrounded by text like “24 Apr 2022 – Relegation”
DATE_PAT = re.compile(r"(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{4})")

def extract_date_from_text(raw: str) -> Optional[str]:
    """
    Return the matched 'DD Mon YYYY' or 'DD Month YYYY' substring from any header line,
    ignoring trailing qualifiers (e.g., '– Relegation', '– Play-offs').
    """
    if not raw:
        return None
    raw = raw.strip()
    m = DATE_PAT.search(raw)
    return m.group(0) if m else None

def extract_date_from_row(row) -> Optional[str]:
    try:
        date_el = row.find_element(
            By.XPATH,
            ".//preceding::div[@data-testid='secondary-header'][1]//div[@data-testid='date-header']//div"
        )
        raw = date_el.text.strip()
        # print for debug if you want:
        # print("HEADER RAW:", raw)
        return extract_date_from_text(raw)
    except Exception:
        return None

def _parse_date(d: Optional[str]):
    """
    Parse either 'DD Mon YYYY' or 'DD Month YYYY'. If there's extra text, DATE_PAT will have
    already reduced it to the matched date substring.
    """
    if not d:
        return None
    s = d.strip()
    # try abbreviated month
    for fmt in ("%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None

# -------------------- Other extraction helpers --------------------
def extract_time(row) -> Optional[str]:
    try:
        el = row.find_element(By.XPATH, ".//div[@data-testid='time-item']//p")
        return el.text.strip()
    except Exception:
        return None

def extract_teams_and_result(row) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    home_name = away_name = None
    home_goals = away_goals = None
    part = row.find_element(By.XPATH, ".//div[@data-testid='event-participants']")

    anchors = part.find_elements(By.XPATH, ".//a[.//p[contains(@class,'participant-name')]]")
    if len(anchors) >= 2:
        try:  home_name = anchors[0].find_element(By.XPATH, ".//p[contains(@class,'participant-name')]").text.strip()
        except Exception: pass
        try:  away_name = anchors[1].find_element(By.XPATH, ".//p[contains(@class,'participant-name')]").text.strip()
        except Exception: pass

        # right-edge scores
        try:
            hg = anchors[0].find_element(By.XPATH, ".//div[contains(@class,'ml-auto') and contains(@class,'font-bold')]").text.strip()
            if hg: home_goals = hg
        except Exception: pass
        try:
            ag = anchors[1].find_element(By.XPATH, ".//div[contains(@class,'ml-auto') and contains(@class,'font-bold')]").text.strip()
            if ag: away_goals = ag
        except Exception: pass

    # fallback center tiny “1–0”
    if home_goals is None or away_goals is None:
        try:
            center = part.find_element(By.XPATH, ".//div[contains(@class,'text-gray-dark') and contains(@class,'relative')]//div[contains(@class,'gap-1')]")
            raw = center.get_attribute("textContent") or ""
            m = re.search(r"(\d+)\s*[–-]\s*(\d+)", raw)
            if m:
                home_goals = home_goals or m.group(1)
                away_goals = away_goals or m.group(2)
        except Exception:
            pass

    result = f"{home_goals}-{away_goals}" if (home_goals is not None and away_goals is not None) else None
    return home_name, away_name, result

def extract_odds_and_bs(row) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    odd_1 = odd_x = odd_2 = None
    bs_value = None
    try:
        odd_cells = row.find_elements(By.XPATH, ".//following-sibling::div[contains(@data-testid,'odd-container')][position()<=3]")
        def get_odd(cell):
            try:
                p = cell.find_element(By.XPATH, ".//p[contains(@data-testid,'odd-container')]")
                return p.text.strip()
            except Exception:
                return None
        if len(odd_cells) >= 1: odd_1 = get_odd(odd_cells[0])
        if len(odd_cells) >= 2: odd_x = get_odd(odd_cells[1])
        if len(odd_cells) >= 3: odd_2 = get_odd(odd_cells[2])
    except Exception:
        pass
    try:
        bs_el = row.find_element(By.XPATH, ".//following-sibling::div[@data-testid='bookies-amount-item']//div[contains(@class,'height-content')]")
        bs_value = bs_el.text.strip()
    except Exception:
        pass
    return odd_1, odd_x, odd_2, bs_value

# -------------------- Data model --------------------
@dataclass
class MatchRow:
    country: str
    league: str
    season_start: int
    page: int
    date_str: Optional[str]
    time_str: Optional[str]
    home_team: Optional[str]
    away_team: Optional[str]
    result: Optional[str]
    odd_1: Optional[str]
    odd_X: Optional[str]
    odd_2: Optional[str]
    bets: Optional[str]

# -------------------- Season helpers --------------------
def infer_season_start(league: LeagueConfig, d: Optional[date]) -> int:
    if d is None:
        d = datetime.today().date()
    if league.kind == "single_year":
        return d.year
    # two-year: Jul–Dec -> start this year; Jan–Jun -> start previous
    return d.year if d.month >= 7 else d.year - 1

# -------------------- Scrape page (results; season fixed) --------------------
def collect_rows_on_page(driver, country: str, league: str, season_start: int, page_num: int) -> List[MatchRow]:
    rows: List[MatchRow] = []
    scroll_to_bottom_until_stable(driver, expected_rows_per_page=50, min_stable_checks=2)

    row_boxes = driver.find_elements(
        By.XPATH,
        "//div[@data-testid='game-row']/ancestor::div[contains(@class,'group') and contains(@class,'flex')]"
    )
    for box in row_boxes:
        try:
            date_s = extract_date_from_row(box)           # <-- sanitized now
            tm = extract_time(box)
            home, away, result = extract_teams_and_result(box)
            o1, ox, o2, bs = extract_odds_and_bs(box)

            print(f"[{country}][{league}][{season_start}] p{page_num} | {date_s or '?'} {tm or '?'} | "
                  f"{home or '?'} vs {away or '?'} -> {result or '?'} | 1:{o1 or '?'} X:{ox or '?'} 2:{o2 or '?'} | bets:{bs or '?'}")

            rows.append(MatchRow(
                country=country, league=league, season_start=season_start, page=page_num,
                date_str=date_s, time_str=tm, home_team=home, away_team=away, result=result,
                odd_1=o1, odd_X=ox, odd_2=o2, bets=bs
            ))
        except StaleElementReferenceException:
            continue
        except Exception:
            continue
    return rows

# -------------------- Scrape page (next matches; season inferred per row) --------------------
def collect_rows_on_page_dynamic_season(driver, league_cfg: LeagueConfig, page_num: int) -> List[MatchRow]:
    rows: List[MatchRow] = []
    scroll_to_bottom_until_stable(driver, expected_rows_per_page=50, min_stable_checks=2)

    row_boxes = driver.find_elements(
        By.XPATH,
        "//div[@data-testid='game-row']/ancestor::div[contains(@class,'group') and contains(@class,'flex')]"
    )
    for box in row_boxes:
        try:
            date_s = extract_date_from_row(box)          # sanitized
            parsed_date = _parse_date(date_s)
            season_start = infer_season_start(league_cfg, parsed_date)

            tm = extract_time(box)
            home, away, result = extract_teams_and_result(box)  # likely None
            o1, ox, o2, bs = extract_odds_and_bs(box)

            print(f"[{league_cfg.country}][{league_cfg.league_name}][{season_start}] p{page_num} | {date_s or '?'} {tm or '?'} | "
                  f"{home or '?'} vs {away or '?'} -> {result or '-'} | 1:{o1 or '-'} X:{ox or '-'} 2:{o2 or '-'} | bets:{bs or '-'}")

            rows.append(MatchRow(
                country=league_cfg.country, league=league_cfg.league_name,
                season_start=season_start, page=page_num,
                date_str=date_s, time_str=tm, home_team=home, away_team=away, result=result,
                odd_1=o1, odd_X=ox, odd_2=o2, bets=bs
            ))
        except StaleElementReferenceException:
            continue
        except Exception:
            continue
    return rows

# -------------------- Postgres helpers --------------------
def _parse_time(t: Optional[str]):
    if not t: return None
    t = t.strip()
    for fmt in ("%H:%M", "%H.%M"):
        try:
            return datetime.strptime(t, fmt).time()
        except ValueError:
            continue
    return None

def _to_decimal(s: Optional[str]):
    if not s: return None
    try: return Decimal(s)
    except Exception: return None

def _to_int(s: Optional[str]):
    if not s: return None
    try: return int(str(s).strip())
    except Exception: return None

def build_insert_values(rows: List[MatchRow]) -> List[Tuple]:
    vals = []
    for r in rows:
        d = _parse_date(r.date_str)          # robust date parsing
        if d is None:
            # Safety: skip rows whose date we can't parse (avoid NOT NULL violation)
            print(f"!! SKIP (no date): [{r.country}][{r.league}] {r.date_str} {r.time_str} {r.home_team} vs {r.away_team}")
            continue
        vals.append((
            r.country,
            r.league,
            int(r.season_start) if r.season_start is not None else None,
            d,
            _parse_time(r.time_str),
            (r.home_team or None),
            (r.away_team or None),
            (r.result or None),
            None,  # half_first
            None,  # half_second
            _to_decimal(r.odd_1),
            _to_decimal(r.odd_X),
            _to_decimal(r.odd_2),
            _to_int(r.bets),
        ))
    return vals

def insert_rows(conn, values: List[Tuple]):
    if not values:
        return
    sql = f"""
    INSERT INTO {TABLE}
    (country, league, season, "date", "time", home_team, away_team, result, half_first, half_second, odd_1, "odd_X", odd_2, bets)
    VALUES %s
    ON CONFLICT (country, league, season, "date", "time", home_team, away_team) DO NOTHING;
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, values)
    conn.commit()

# -------------------- Orchestration --------------------
SCRAPE_RESULTS = True
SCRAPE_NEXT    = True

def scrape_results_for_league(conn, driver, league: LeagueConfig):
    for start_year in league.seasons:
        url = league.results_url(start_year)
        print(f"\n=== RESULTS • {league.country.upper()} • {league.league_name} • {start_year} ===")
        go_to_url(driver, url)

        total_pages = get_total_pages(driver)
        if total_pages is None:
            page_idx = 1
            while True:
                print(f"-- Page {page_idx}")
                rows = collect_rows_on_page(driver, league.country, league.league_name, start_year, page_idx)
                insert_rows(conn, build_insert_values(rows))
                if not click_next_page(driver):
                    break
                page_idx += 1
        else:
            for p in range(1, total_pages + 1):
                print(f"-- Page {p}/{total_pages}")
                rows = collect_rows_on_page(driver, league.country, league.league_name, start_year, p)
                insert_rows(conn, build_insert_values(rows))
                if p < total_pages and not click_next_page(driver):
                    print("Next not found/disabled early; stopping this season.")
                    break

def scrape_next_for_league(conn, driver, league: LeagueConfig):
    url = league.next_url()
    print(f"\n=== NEXT MATCHES • {league.country.upper()} • {league.league_name} ===")
    go_to_url(driver, url)
    # Next-matches pages often aren’t paginated; if they are, logic can be extended similarly.
    rows = collect_rows_on_page_dynamic_season(driver, league, page_num=1)
    insert_rows(conn, build_insert_values(rows))

def main(headless=True):
    conn = psycopg2.connect(**DB_CONFIG)
    driver = make_driver(headless=headless)
    try:
        for lg in LEAGUES:
            if SCRAPE_RESULTS:
                scrape_results_for_league(conn, driver, lg)
            if SCRAPE_NEXT:
                scrape_next_for_league(conn, driver, lg)
    finally:
        try: driver.quit()
        except Exception: pass
        try: conn.close()
        except Exception: pass

if __name__ == "__main__":
    main(headless=True)
