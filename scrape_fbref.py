import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup, Comment
import time

def get_html_content(url):
    """
    Fetches the HTML content of a given URL.
    Includes headers to mimic a web browser.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.text
    else:
        print(f"Error fetching {url}: Status Code {response.status_code}")
        return None

def scrape_table(html_content, table_id):
    """
    Scrapes a single table from the HTML content and returns a DataFrame.
    This version correctly extracts tables even if they are within HTML comments inside a div.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        target_table_html = None
        
        # First, try to find the table directly
        table_element = soup.find('table', {'id': table_id})
        
        # If not found directly, search within comments for the div container
        if not table_element:
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                # Check if the comment contains the div with the target ID
                if f'div_{table_id}' in comment: # Notice 'div_' prefix for the container
                    # Parse the content of the comment as HTML
                    comment_soup = BeautifulSoup(comment, 'html.parser')
                    # Find the actual table element inside that parsed comment content
                    table_element = comment_soup.find('table')
                    if table_element:
                        break # Found it, exit comment loop
        
        if table_element:
            # Now, pass the *actual table HTML* to pandas
            df = pd.read_html(StringIO(str(table_element)), header=[0, 1])[0]
            # Drop the redundant header rows if they exist in the table body (e.g., "Squad" repeated)
            if 'Squad' in df.columns:
                df = df.drop(df[df['Squad'] == 'Squad'].index)
            return df
        else:
            print(f"No table with id '{table_id}' found in HTML or comments.")
            return None
    except Exception as e: # Broaden exception to catch any parsing issues
        print(f"Error in scrape_table for ID '{table_id}': {e}")
        return None

def clean_and_prefix_df(df, prefix):
    """
    Flattens multi-level headers and applies a prefix to ALL columns except 'Squad'.
    This ensures distinct names for metrics that appear in both offensive and defensive tables.
    """
    if df is not None:
        flattened_cols = []
        for col_tuple in df.columns:
            # Check if it's a multi-index tuple
            if isinstance(col_tuple, tuple):
                top_level, sub_level = col_tuple
                if "Unnamed" in top_level: # e.g., ('Unnamed: 0_level_0', 'Squad')
                    flattened_cols.append(sub_level)
                else:
                    flattened_cols.append(f"{top_level}_{sub_level}")
            else: # Already a single level header (e.g., 'Squad' from raw pd.read_html)
                flattened_cols.append(col_tuple)
        df.columns = flattened_cols

        new_columns = []
        for col in df.columns:
            if col == 'Squad': # 'Squad' is the unique identifier for merging
                new_columns.append(col)
            else:
                new_columns.append(f"{prefix}_{col}")
        df.columns = new_columns
        return df
    return None


def get_and_process_stats(url, columns_to_drop):
    """
    Fetches, scrapes, cleans, and merges data from both offensive and defensive tables.
    """
    html_content = get_html_content(url)
    if not html_content:
        return None

    df_offensive_raw = scrape_table(html_content, 'stats_squads_standard_for')
    df_offensive = clean_and_prefix_df(df_offensive_raw.copy(), 'Offensive')

    df_defensive_raw = scrape_table(html_content, 'stats_squads_standard_against')
    df_defensive = clean_and_prefix_df(df_defensive_raw.copy(), 'Defensive')
    
    if df_offensive is None or df_defensive is None:
        print("One or more tables could not be scraped. Returning None.")
        return None

    if 'Squad' not in df_offensive.columns or 'Squad' not in df_defensive.columns:
        print("Error: 'Squad' column missing from one of the DataFrames. Cannot merge.")
        return None

    df_combined = pd.merge(df_offensive, 
                           df_defensive, 
                           on='Squad', 
                           how='left')
    
    # STRICTLY DROPPING ONLY THE SPECIFIED COLUMNS
    df_combined = df_combined.drop(columns=[col for col in columns_to_drop if col in df_combined.columns], errors='ignore')

    df_combined = df_combined.reset_index(drop=True)

    return df_combined

# --- Main script execution starts here ---

# The ONLY columns to drop, as strictly specified:
columns_to_drop = [
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


# A list of years to scrape
years = range(2017, 2025)

# The base URL, with a placeholder for the year.
base_url = "https://fbref.com/en/comps/9/{year}-{next_year}/Premier-League-Stats"

for year in years:
    next_year = year + 1
    url = base_url.format(year=year, next_year=next_year)
    print(f"\n--- Scraping data for the {year}-{next_year} season from {url} ---")
    
    team_stats_df = get_and_process_stats(url, columns_to_drop)

    if team_stats_df is not None:
        print(f"DataFrame for {year}-{next_year} season has {len(team_stats_df.columns)} columns.")
        print("Final DataFrame head:")
        print(team_stats_df.head())
        print("All columns in the final DataFrame:")
        print(team_stats_df.columns.tolist())
        
        output_path = f'footballprediction/fbref_data/scrapped_team_stats_{year}.csv'
        team_stats_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved the scraped data to '{output_path}'")
    else:
        print(f"Could not scrape data for the {year}-{next_year} season.")

    time.sleep(15) 
