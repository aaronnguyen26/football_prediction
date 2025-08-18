import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup, Comment
import time
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.4472.124 Safari/537.36"
}

BASE_URL = "https://fbref.com/en/comps/9/{year}-{next_year}/Premier-League-Stats"

COLUMNS_TO_DROP = [
    'Offensive_Playing Time_MP',
    'Offensive_Playing Time_Starts',
    'Offensive_Playing Time_Min',
    'Offensive_Playing Time_90s',
    'Offensive_# Pl',
    'Offensive_Age',
    'Defensive_Playing Time_MP',
    'Defensive_Playing Time_Starts',
    'Defensive_Playing Time_Min',
    'Defensive_Playing Time_90s',
    'Defensive_# Pl',
    'Defensive_Age'
]

# Use full range or test with one year: [2018]
YEARS = range(2017, 2025)

# Ensure output directory exists
OUTPUT_DIR = 'footballprediction/fbref_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# TEAM NAME NORMALIZATION
# -----------------------------
def normalize_squad_name(name):
    """Standardize team names across seasons for reliable merging."""
    if pd.isna(name) or name == '' or name is None:
        return "Unknown"

    name = str(name).strip()

    mapping = {
        'Tottenham Hotspur': 'Tottenham',
        'Newcastle United': 'Newcastle Utd',
        'Manchester United': 'Man Utd',
        'Manchester City': 'Man City',
        'West Ham United': 'West Ham',
        'Leicester City': 'Leicester',
        'Leeds United': 'Leeds',
        'Sheffield United': 'Sheffield Utd',
        'Wolverhampton Wanderers': 'Wolves',
        'Brighton and Hove Albion': 'Brighton',
        'Norwich City': 'Norwich',
        'AFC Bournemouth': 'Bournemouth',
        'Stoke City': 'Stoke',
        'Huddersfield Town': 'Huddersfield',
        'West Bromwich Albion': 'West Brom',
        'Swansea City': 'Swansea',
        'Cardiff City': 'Cardiff',
        'Derby County': 'Derby',
        'Middlesbrough': 'Middlesbrough',
        'Aston Villa': 'Aston Villa',
        'Fulham': 'Fulham',
        'Brentford': 'Brentford',
        'Ipswich Town': 'Ipswich',
        'Crystal Palace': 'Crystal Palace',
        'Arsenal': 'Arsenal',
        'Chelsea': 'Chelsea',
        'Liverpool': 'Liverpool',
        'Everton': 'Everton',
        'Southampton': 'Southampton',
        'Burnley': 'Burnley',
        'Watford': 'Watford',
    }

    return mapping.get(name, name)


# -----------------------------
# HELPER: Fetch HTML
# -----------------------------
def get_html_content(url):
    """Fetch HTML with error handling and retries."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            print(f"✅ Successfully fetched: {url}")
            return response.text
        else:
            print(f"❌ HTTP {response.status_code} for {url}")
            return None
    except Exception as e:
        print(f"⚠️ Request failed for {url}: {e}")
        return None


# -----------------------------
# HELPER: Parse Table Safely
# -----------------------------
def safe_read_html(table_html):
    """Safely parse table HTML using pandas."""
    try:
        df = pd.read_html(StringIO(str(table_html)), header=[0, 1])[0]
        # Drop duplicate header rows (e.g., 'Squad' repeated in body)
        if 'Squad' in df.columns:
            df = df[df['Squad'] != 'Squad']
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"❌ Failed to parse table with pandas: {e}")
        return None


# -----------------------------
# MAIN: Scrape Table (Robust)
# -----------------------------
def scrape_table(html_content, table_id):
    """
    Robustly scrape a table by ID, even if hidden in HTML comments.
    Falls back to structural search if ID is missing.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # 1. Try direct lookup
        table = soup.find('table', {'id': table_id})
        if table:
            print(f"  ✅ Found '{table_id}' directly")
            return safe_read_html(table)

        # 2. Search in HTML comments
        comments = soup.find_all(string=lambda x: isinstance(x, Comment))
        for comment in comments:
            # Look for div container
            if f'div_{table_id}' in comment:
                print(f"  🔍 Found '{table_id}' in comment, parsing...")
                comment_soup = BeautifulSoup(comment, 'html.parser')
                table = comment_soup.find('table', {'id': table_id})
                if table:
                    return safe_read_html(table)

            # Fallback: look for any table with 'Squad' column
            if 'Squad' in comment and ('GA' in comment or 'Gls' in comment):
                print(f"  ⚠️ Fallback: parsing comment with 'Squad' for '{table_id}'")
                comment_soup = BeautifulSoup(comment, 'html.parser')
                table = comment_soup.find('table')
                if table:
                    df = safe_read_html(table)
                    if df is not None and 'Squad' in df.columns:
                        return df

        print(f"  ❌ Table '{table_id}' not found")
        return None

    except Exception as e:
        print(f"  ❌ Error scraping '{table_id}': {e}")
        return None


# -----------------------------
# CLEAN & PREFIX COLUMNS (No Prefix Yet)
# -----------------------------
def clean_and_prefix_df(df, prefix_label):
    """Flatten multi-index columns and return clean DataFrame (no prefix yet)."""
    if df is None or df.empty:
        print(f"⚠️ Empty or None DataFrame for '{prefix_label}'")
        return None

    # Flatten multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        for col in df.columns:
            if col[0].startswith('Unnamed'):
                cols.append(col[1])
            else:
                cols.append(f"{col[0]}_{col[1]}")
        df.columns = cols

    # Ensure 'Squad' exists
    if 'Squad' not in df.columns:
        print(f"❌ 'Squad' column missing in {prefix_label} DataFrame. Columns: {df.columns.tolist()}")
        return None

    return df


# -----------------------------
# GET & PROCESS STATS (Fixed: Remove 'vs ' and Normalize)
# -----------------------------
def get_and_process_stats(url, columns_to_drop):
    """Fetch, scrape, clean, and merge offensive and defensive stats."""
    html = get_html_content(url)
    if not html:
        return None

    print("🔍 Scraping offensive stats (team's own performance)...")
    df_off_raw = scrape_table(html, 'stats_squads_standard_for')
    df_off = clean_and_prefix_df(df_off_raw, 'Offensive')

    print("🔍 Scraping defensive stats (opponents' performance vs team)...")
    df_def_raw = scrape_table(html, 'stats_squads_standard_against')
    df_def = clean_and_prefix_df(df_def_raw, 'Defensive')

    # Check for success
    if df_off is None:
        print("❌ Failed to get offensive stats")
        return None
    if df_def is None:
        print("❌ Failed to get defensive stats")
        return None

    # --- 🔥 CRITICAL FIX 1: Remove 'vs ' prefix from defensive Squad names ---
    df_def['Squad'] = df_def['Squad'].astype(str).str.replace(r'^vs\s+', '', regex=True).str.strip()

    # --- 🔥 CRITICAL FIX 2: Normalize team names BEFORE merge ---
    df_off['Squad'] = df_off['Squad'].apply(normalize_squad_name)
    df_def['Squad'] = df_def['Squad'].apply(normalize_squad_name)

    # Debug: Show team names
    print(f"  🧾 Offensive teams: {sorted(df_off['Squad'].unique())}")
    print(f"  🧾 Defensive teams: {sorted(df_def['Squad'].unique())}")

    # Now merge
    print("📊 Merging offensive and defensive stats...")
    df_combined = pd.merge(df_off, df_def, on='Squad', how='inner')

    if df_combined.empty:
        print("❌ Merge failed: No matching 'Squad' names between tables")
        missing = set(df_off['Squad']) - set(df_def['Squad'])
        extra = set(df_def['Squad']) - set(df_off['Squad'])
        if missing: print(f"🔍 Missing in defensive: {missing}")
        if extra: print(f"🔍 Missing in offensive: {extra}")
        return None

    print(f"✅ Merge successful: {len(df_combined)} teams matched")

    # --- Apply prefixes after merge ---
    # Identify which columns came from which table
    suffix_map = {}
    for col in df_combined.columns:
        if col in df_off.columns and col != 'Squad':
            suffix_map[col] = 'Offensive'
        elif col in df_def.columns and col != 'Squad':
            suffix_map[col] = 'Defensive'

    df_combined.rename(columns={col: f"{suffix}_{col}" for col, suffix in suffix_map.items()}, inplace=True)

    # Drop unnecessary columns
    cols_dropped = [col for col in columns_to_drop if col in df_combined.columns]
    df_combined = df_combined.drop(columns=cols_dropped, errors='ignore')
    if cols_dropped:
        print(f"🗑️  Dropped {len(cols_dropped)} columns (e.g., playing time, age)")

    return df_combined.reset_index(drop=True)


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting FBref Premier League Scraper\n")

    for year in YEARS:
        next_year = year + 1
        url = BASE_URL.format(year=year, next_year=next_year)
        output_file = f"{OUTPUT_DIR}/scrapped_team_stats_{year}.csv"

        print(f"\n" + "="*60)
        print(f"📌 SCRAPING {year}-{next_year} SEASON")
        print(f"🔗 {url}")
        print("="*60)

        df_season = get_and_process_stats(url, COLUMNS_TO_DROP)

        if df_season is not None:
            df_season.to_csv(output_file, index=False)
            print(f"✅ SUCCESS: Saved {len(df_season)} teams to {output_file}")

            # Preview
            print(f"\n📋 Preview of merged data ({year}-{next_year}):")
            preview_cols = ['Squad'] + \
                           [c for c in df_season.columns if 'Poss' in c or 'Gls' in c or 'xG' in c][:4]
            print(df_season[preview_cols].head())
        else:
            print(f"❌ FAILED to scrape {year}-{next_year}")

        # Be respectful
        print(f"⏳ Waiting 15 seconds before next season...")
        time.sleep(15)

    print("\n🎉 Scraping complete! All team stats saved.")