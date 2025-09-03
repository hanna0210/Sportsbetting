import time
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from decimal import Decimal
from datetime import datetime

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    StaleElementReferenceException,
    ElementClickInterceptedException,
)

import psycopg2
from psycopg2.extras import execute_values

# -------------------- DB CONFIG --------------------
DB_CONFIG = {
    "host":     "127.0.0.1",
    "port":     5432,
    "dbname":   "sportsbetting",
    "user":     "postgres",
    "password": "q1w2e3",
}
SCHEMA = "bettingschema"
TABLE  = "odds"

# -------------------- League configuration --------------------
@dataclass
class LeagueConfig:
    country: str              # e.g., "brazil"
    base: str                 # e.g., "https://www.oddsportal.com/football/brazil/"
    kind: str                 # "single_year" or "two_year"
    comp_slug: str            # e.g., "serie-a" / "premier-league" / "laliga"
    seasons: List[int]        # single_year: [2021...]; two_year: start years [2021,2022,...]
    # Optional per-league overrides for tricky slugs
    special_slugs: Dict[int, str] = None        # maps start_year -> "path-ending/"

    def start_url(self, start_year: int) -> str:
        """
        Build the results path for a given (start) year.
        - single_year: comp-{year}/results/
        - two_year: comp-{start}-{start+1}/results/
        - current two-year (e.g., 2025-2026): comp/results/ (handled by special_slugs or rule below)
        - Brazil special cases handled via special_slugs
        """
        # special cases win
        if self.special_slugs and start_year in self.special_slugs:
            return self.base + self.special_slugs[start_year]

        if self.kind == "single_year":
            return f"{self.base}{self.comp_slug}-{start_year}/results/"

        # two-year:
        if start_year >= 2025:
            # current no-year style for 2025-2026
            return f"{self.base}{self.comp_slug}/results/"
        return f"{self.base}{self.comp_slug}-{start_year}-{start_year+1}/results/"

# --- Define leagues ---
BRAZIL = LeagueConfig(
    country="brazil",
    base="https://www.oddsportal.com/football/brazil/",
    kind="single_year",
    comp_slug="serie-a",
    seasons=[2021, 2022, 2023, 2024, 2025],
    special_slugs={
        2024: "serie-a-betano-2024/results/",
        2025: "serie-a-betano/results/",
    },
)

ENGLAND = LeagueConfig(
    country="england",
    base="https://www.oddsportal.com/football/england/",
    kind="two_year",
    comp_slug="premier-league",
    seasons=[2021, 2022, 2023, 2024, 2025],  # 2025 -> 2025-2026 (no-year slug)
)

SPAIN = LeagueConfig(
    country="spain",
    base="https://www.oddsportal.com/football/spain/",
    kind="two_year",
    comp_slug="laliga",
    seasons=[2021, 2022, 2023, 2024, 2025],  # 2025 -> 2025-2026 (no-year slug)
)

LEAGUES = [BRAZIL, ENGLAND, SPAIN]

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

def go_to_first_page(driver, league: LeagueConfig, start_year: int):
    url = league.start_url(start_year)
    driver.get(url)
    wait_for_results_table(driver)
    close_popups(driver)

# Render everything before scraping & clicking Next
def _row_count(driver) -> int:
    return len(driver.find_elements(
        By.XPATH,
        "//div[@data-testid='game-row']/ancestor::div[contains(@class,'group') and contains(@class,'flex')]"
    ))

def scroll_to_bottom_until_stable(
    driver,
    *,
    expected_rows_per_page: int = 50,
    min_stable_checks: int = 2,
    max_loops: int = 40,
    pause: float = 0.25
) -> int:
    stable = 0
    prev_h = -1
    prev_rows = -1
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(0.2)

    for _ in range(max_loops):
        driver.execute_script("window.scrollBy(0, Math.max(700, window.innerHeight*0.9));")
        time.sleep(pause)

        at_bottom = driver.execute_script(
            "return (window.innerHeight + window.scrollY) >= (document.body.scrollHeight - 4);"
        )
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
    if not links:
        return None
    return links[-1]  # bottom pager

def click_next_page(driver) -> bool:
    scroll_to_bottom_until_stable(driver)
    close_popups(driver)

    btn = locate_next_button(driver)
    if not btn:
        return False

    driver.execute_script("""
        const el = arguments[0];
        el.scrollIntoView({block: 'center', inline: 'nearest'});
        window.scrollBy(0, -140);
    """, btn)
    time.sleep(0.1)

    for _ in range(6):
        covered = driver.execute_script("""
            const el = arguments[0];
            const r = el.getBoundingClientRect();
            const x = Math.floor(r.left + r.width/2);
            const y = Math.floor(r.top + r.height/2);
            const top = document.elementFromPoint(x, y);
            return !(top && (top === el || el.contains(top)));
        """, btn)
        if not covered:
            break
        driver.execute_script("window.scrollBy(0, 40);")
        time.sleep(0.05)

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
    links = driver.find_elements(By.XPATH, xp)
    nums = []
    for el in links:
        t = (el.text or "").strip()
        if t.isdigit():
            nums.append(int(t))
    return max(nums) if nums else None

# -------------------- Extraction --------------------
def extract_date_from_row(row) -> Optional[str]:
    try:
        date_el = row.find_element(
            By.XPATH,
            ".//preceding::div[@data-testid='secondary-header'][1]"
            "//div[@data-testid='date-header']//div"
        )
        return date_el.text.strip()
    except Exception:
        return None

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
        try:
            home_name = anchors[0].find_element(By.XPATH, ".//p[contains(@class,'participant-name')]").text.strip()
        except Exception:
            pass
        try:
            away_name = anchors[1].find_element(By.XPATH, ".//p[contains(@class,'participant-name')]").text.strip()
        except Exception:
            pass

        # Right-edge bold numbers (desktop)
        try:
            hg = anchors[0].find_element(By.XPATH, ".//div[contains(@class,'ml-auto') and contains(@class,'font-bold')]").text.strip()
            if hg:
                home_goals = hg
        except Exception:
            pass
        try:
            ag = anchors[1].find_element(By.XPATH, ".//div[contains(@class,'ml-auto') and contains(@class,'font-bold')]").text.strip()
            if ag:
                away_goals = ag
        except Exception:
            pass

    # Center fallback "1–0"
    if home_goals is None or away_goals is None:
        try:
            center = part.find_element(
                By.XPATH,
                ".//div[contains(@class,'text-gray-dark') and contains(@class,'relative')]//div[contains(@class,'gap-1')]"
            )
            raw = center.get_attribute("textContent") or ""
            m = re.search(r"(\d+)\s*[–-]\s*(\d+)", raw)
            if m:
                home_goals = home_goals or m.group(1)
                away_goals = away_goals or m.group(2)
        except Exception:
            pass

    result = f"{home_goals}-{away_goals}" if (home_goals is not None and away_goals is not None) else None
    return home_name, away_name, result

def extract_odds_and_bs(row):
    odd_1 = odd_x = odd_2 = None
    bs_value = None

    try:
        odd_cells = row.find_elements(
            By.XPATH, ".//following-sibling::div[contains(@data-testid,'odd-container')][position()<=3]"
        )
        def get_odd(cell):
            try:
                p = cell.find_element(By.XPATH, ".//p[contains(@data-testid,'odd-container')]")
                return p.text.strip()
            except Exception:
                return None

        if len(odd_cells) >= 1:
            odd_1 = get_odd(odd_cells[0])
        if len(odd_cells) >= 2:
            odd_x = get_odd(odd_cells[1])
        if len(odd_cells) >= 3:
            odd_2 = get_odd(odd_cells[2])
    except Exception:
        pass

    try:
        bs_el = row.find_element(
            By.XPATH,
            ".//following-sibling::div[@data-testid='bookies-amount-item']//div[contains(@class,'height-content')]"
        )
        bs_value = bs_el.text.strip()
    except Exception:
        pass

    return odd_1, odd_x, odd_2, bs_value

# -------------------- Data model --------------------
@dataclass
class MatchRow:
    season_start: int           # store start year only (e.g., 2021 for 2021–2022)
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

# -------------------- Scrape page --------------------
def collect_rows_on_page(driver, season_start: int, page_num: int) -> List[MatchRow]:
    rows: List[MatchRow] = []
    scroll_to_bottom_until_stable(driver, expected_rows_per_page=50, min_stable_checks=2)

    row_boxes = driver.find_elements(
        By.XPATH,
        "//div[@data-testid='game-row']/ancestor::div[contains(@class,'group') and contains(@class,'flex')]"
    )
    for box in row_boxes:
        try:
            date = extract_date_from_row(box)
            tm = extract_time(box)
            home, away, result = extract_teams_and_result(box)
            o1, ox, o2, bs = extract_odds_and_bs(box)
            rows.append(
                MatchRow(
                    season_start=season_start,
                    page=page_num,
                    date_str=date,
                    time_str=tm,
                    home_team=home,
                    away_team=away,
                    result=result,
                    odd_1=o1,
                    odd_X=ox,
                    odd_2=o2,
                    bets=bs,
                )
            )
        except StaleElementReferenceException:
            continue
        except Exception:
            continue

    return rows

# -------------------- Postgres helpers --------------------
def _parse_date(d: Optional[str]):
    if not d:
        return None
    return datetime.strptime(d.strip(), "%d %b %Y").date()

def _parse_time(t: Optional[str]):
    if not t:
        return None
    t = t.strip()
    for fmt in ("%H:%M", "%H.%M"):
        try:
            return datetime.strptime(t, fmt).time()
        except ValueError:
            continue
    return None

def _to_decimal(s: Optional[str]):
    if s is None or s == "":
        return None
    try:
        return Decimal(s)
    except Exception:
        return None

def _to_int(s: Optional[str]):
    if s is None or s == "":
        return None
    try:
        return int(str(s).strip())
    except Exception:
        return None

def build_insert_values(rows: List[MatchRow]) -> List[Tuple]:
    values = []
    for r in rows:
        values.append((
            r.season_start,                   # season INTEGER = start year
            _parse_date(r.date_str),
            _parse_time(r.time_str),
            (r.home_team or None),
            (r.away_team or None),
            (r.result or None),
            None,  # half_first (list page doesn't have it)
            None,  # half_second
            _to_decimal(r.odd_1),
            _to_decimal(r.odd_X),
            _to_decimal(r.odd_2),
            _to_int(r.bets),
        ))
    return values

def insert_rows(conn, values: List[Tuple]):
    if not values:
        return
    sql = f"""
    INSERT INTO {SCHEMA}.{TABLE}
    (season, "date", "time", home_team, away_team, result, half_first, half_second, odd_1, "odd_X", odd_2, bets)
    VALUES %s
    ON CONFLICT (season, "date", "time", home_team, away_team) DO NOTHING;
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, values)
    conn.commit()

# -------------------- Main --------------------
def main(headless=True):
    conn = psycopg2.connect(**DB_CONFIG)
    driver = make_driver(headless=headless)
    try:
        for league in LEAGUES:
            for start_year in league.seasons:
                print(f"[{league.country}] season start {start_year} → open…")
                go_to_first_page(driver, league, start_year)

                total_pages = get_total_pages(driver)
                if total_pages is None:
                    page_idx = 1
                    while True:
                        print(f"  Page {page_idx}")
                        rows = collect_rows_on_page(driver, start_year, page_idx)
                        print(f"    Collected {len(rows)} rows")
                        insert_rows(conn, build_insert_values(rows))
                        if not click_next_page(driver):
                            break
                        page_idx += 1
                else:
                    for p in range(1, total_pages + 1):
                        print(f"  Page {p}/{total_pages}")
                        rows = collect_rows_on_page(driver, start_year, p)
                        print(f"    Collected {len(rows)} rows")
                        insert_rows(conn, build_insert_values(rows))
                        if p < total_pages:
                            if not click_next_page(driver):
                                print("    Next missing early; stopping this season.")
                                break
    finally:
        try:
            driver.quit()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    main(headless=True)
