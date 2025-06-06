from pathlib import Path
from typing import List, Tuple
import pandas as pd
import re

def process_season_game_pair(season_file: str, game_file: str, season_year: int, game_year: int) -> pd.DataFrame:
    """
    Process a single season-game file pair and return merged dataset
    
    Args:
        season_file: Path to season stats CSV (e.g., "cfb19.csv")
        game_file: Path to games CSV (e.g., "games2020.csv") 
        season_year: Year of season stats (e.g., 2019)
        game_year: Year of games (e.g., 2020)
    
    Returns:
        DataFrame with merged season stats and game data
    """
    
    # Load data
    print(f"Processing {season_file} -> {game_file}")
    season_data = pd.read_csv(season_file)
    game_data = pd.read_csv(game_file)
    
    # Process season data
    if "Win-Loss" in season_data.columns:
        season_data["Win-Loss"] = season_data["Win-Loss"].astype(str).str.strip()
        split_wins_losses = season_data["Win-Loss"].str.split("-", expand=True)
        season_data["wins"] = pd.to_numeric(split_wins_losses[0], errors="coerce").fillna(0).astype(int)
        season_data["losses"] = pd.to_numeric(split_wins_losses[1], errors="coerce").fillna(0).astype(int)
        season_data = season_data.drop(columns=["Win-Loss"])
    
    # Filter out postseason games
    game_data = game_data[game_data["season_type"] != "Postseason"]
    
    # Clean team names using mapping
    name_mapping = {
        "Air Force": "Air Force (Mountain West)",
        "Akron": "Akron (MAC)",
        "Alabama": "Alabama (SEC)",
        "Appalachian State": "App State (Sun Belt)",
        "Arizona": "Arizona (Pac-12)",
        "Arizona State": "Arizona St. (Pac-12)",
        "Arkansas": "Arkansas (SEC)",
        "Arkansas State": "Arkansas St. (Sun Belt)",
        "Army": "Army West Point (FBS Independent)",
        "Auburn": "Auburn (SEC)",
        "Ball State": "Ball St. (MAC)",
        "Baylor": "Baylor (Big 12)",
        "Boise State": "Boise St. (Mountain West)",
        "Boston College": "Boston College (ACC)",
        "Bowling Green": "Bowling Green (MAC)",
        "Buffalo": "Buffalo (MAC)",
        "BYU": "BYU (Big 12)",
        "California": "California (Pac-12)",
        "Central Michigan": "Central Mich. (MAC)",
        "Charlotte": "Charlotte (AAC)",
        "Cincinnati": "Cincinnati (Big 12)",
        "Clemson": "Clemson (ACC)",
        "Coastal Carolina": "Coastal Carolina (Sun Belt)",
        "Colorado": "Colorado (Pac-12)",
        "Colorado State": "Colorado St. (Mountain West)",
        "Duke": "Duke (ACC)",
        "East Carolina": "East Carolina (AAC)",
        "Eastern Michigan": "Eastern Mich. (MAC)",
        "FIU": "FIU (CUSA)",
        "Florida Atlantic": "Fla. Atlantic (AAC)",
        "Florida": "Florida (SEC)",
        "Florida State": "Florida St. (ACC)",
        "Fresno State": "Fresno St. (Mountain West)",
        "Georgia Southern": "Ga. Southern (Sun Belt)",
        "Georgia": "Georgia (SEC)",
        "Georgia State": "Georgia St. (Sun Belt)",
        "Georgia Tech": "Georgia Tech (ACC)",
        "Houston": "Houston (Big 12)",
        "Illinois": "Illinois (Big Ten)",
        "Indiana": "Indiana (Big Ten)",
        "Iowa": "Iowa (Big Ten)",
        "Iowa State": "Iowa St. (Big 12)",
        "Kansas": "Kansas (Big 12)",
        "Kansas State": "Kansas St. (Big 12)",
        "Kent State": "Kent St. (MAC)",
        "Kentucky": "Kentucky (SEC)",
        "Louisiana Monroe": "La.-Monroe (Sun Belt)",
        "Liberty": "Liberty (CUSA)",
        "Louisiana": "Louisiana (Sun Belt)",
        "Louisiana Tech": "Louisiana Tech (CUSA)",
        "Louisville": "Louisville (ACC)",
        "LSU": "LSU (SEC)",
        "Massachusetts": "Massachusetts (FBS Independent)",
        "Marshall": "Marshall (Sun Belt)",
        "Maryland": "Maryland (Big Ten)",
        "Memphis": "Memphis (AAC)",
        "Miami": "Miami (FL) (ACC)",
        "Miami (OH)": "Miami (OH) (MAC)",
        "Michigan": "Michigan (Big Ten)",
        "Michigan State": "Michigan St. (Big Ten)",
        "Middle Tennessee": "Middle Tenn. (CUSA)",
        "Minnesota": "Minnesota (Big Ten)",
        "Mississippi State": "Mississippi St. (SEC)",
        "Missouri": "Missouri (SEC)",
        "Navy": "Navy (AAC)",
        "NC State": "NC State (ACC)",
        "Nebraska": "Nebraska (Big Ten)",
        "Nevada": "Nevada (Mountain West)",
        "New Mexico": "New Mexico (Mountain West)",
        "New Mexico State": "New Mexico St. (CUSA)",
        "North Carolina": "North Carolina (ACC)",
        "North Texas": "North Texas (AAC)",
        "Northwestern": "Northwestern (Big Ten)",
        "Notre Dame": "Notre Dame (FBS Independent)",
        "Ohio": "Ohio (MAC)",
        "Ohio State": "Ohio St. (Big Ten)",
        "Oklahoma": "Oklahoma (Big 12)",
        "Oklahoma State": "Oklahoma St. (Big 12)",
        "Ole Miss": "Ole Miss (SEC)",
        "Oregon": "Oregon (Pac-12)",
        "Oregon State": "Oregon St. (Pac-12)",
        "Penn State": "Penn St. (Big Ten)",
        "Pittsburgh": "Pittsburgh (ACC)",
        "Purdue": "Purdue (Big Ten)",
        "Rice": "Rice (AAC)",
        "Rutgers": "Rutgers (Big Ten)",
        "San Diego State": "San Diego St. (Mountain West)",
        "San José State": "San Jose St. (Mountain West)",
        "SMU": "SMU (AAC)",
        "South Alabama": "South Alabama (Sun Belt)",
        "Southern California": "Southern California (Pac-12)",
        "South Carolina": "South Carolina (SEC)",
        "South Florida": "South Fla. (AAC)",
        "Southern Mississippi": "Southern Miss. (Sun Belt)",
        "Stanford": "Stanford (Pac-12)",
        "Syracuse": "Syracuse (ACC)",
        "TCU": "TCU (Big 12)",
        "Temple": "Temple (AAC)",
        "Tennessee": "Tennessee (SEC)",
        "Texas": "Texas (Big 12)",
        "Texas A&M": "Texas A&M (SEC)",
        "Texas State": "Texas St. (Sun Belt)",
        "Texas Tech": "Texas Tech (Big 12)",
        "Toledo": "Toledo (MAC)",
        "Troy": "Troy (Sun Belt)",
        "Tulane": "Tulane (AAC)",
        "Tulsa": "Tulsa (AAC)",
        "UAB": "UAB (AAC)",
        "UCF": "UCF (Big 12)",
        "UCLA": "UCLA (Pac-12)",
        "UNLV": "UNLV (Mountain West)",
        "Utah": "Utah (Pac-12)",
        "Utah State": "Utah St. (Mountain West)",
        "UTEP": "UTEP (CUSA)",
        "UTSA": "UTSA (AAC)",
        "Vanderbilt": "Vanderbilt (SEC)",
        "Virginia": "Virginia (ACC)",
        "Virginia Tech": "Virginia Tech (ACC)",
        "Wake Forest": "Wake Forest (ACC)",
        "Washington": "Washington (Pac-12)",
        "Washington State": "Washington St. (Pac-12)",
        "West Virginia": "West Virginia (Big 12)",
        "Western Kentucky": "Western Ky. (CUSA)",
        "Western Michigan": "Western Mich. (MAC)",
        "Wisconsin": "Wisconsin (Big Ten)",
        "Wyoming": "Wyoming (Mountain West)"
    }
    
    # Apply name mapping
    season_data["Team"] = season_data["Team"].replace(name_mapping)
    game_data["home_team"] = game_data["home_team"].replace(name_mapping)
    game_data["away_team"] = game_data["away_team"].replace(name_mapping)
    
    # Filter games to only include teams in name mapping
    valid_teams = set(name_mapping.values())
    game_data = game_data[
        (game_data["home_team"].isin(valid_teams)) &
        (game_data["away_team"].isin(valid_teams))
    ]
    
    # Standardize column names
    season_data = season_data.rename(columns={'Team': 'team'})
    season_data.columns = [col.lower() for col in season_data.columns]
    
    # Add season identifiers
    season_data['season_stats_year'] = season_year
    game_data['game_year'] = game_year
    
    # Merge season data with game data
    merged = game_data.merge(
        season_data.rename(columns={"wins": "home_wins", "losses": "home_losses"}),
        how="left",
        left_on="home_team",
        right_on="team",
        suffixes=("", "_home")
    ).drop(columns=["team"])
    
    merged = merged.merge(
        season_data.rename(columns={"wins": "away_wins", "losses": "away_losses"}),
        how="left",
        left_on="away_team", 
        right_on="team",
        suffixes=("", "_away")
    ).drop(columns=["team"])
    
    # Handle missing data
    home_cols = [col for col in merged.columns if col.endswith('_home')]
    away_cols = [col for col in merged.columns if col.endswith('_away')]
    
    merged['home_stats_missing'] = merged[home_cols].isnull().any(axis=1).astype(int)
    merged['away_stats_missing'] = merged[away_cols].isnull().any(axis=1).astype(int)
    
    merged[home_cols] = merged[home_cols].fillna(0)
    merged[away_cols] = merged[away_cols].fillna(0)
    
    merged["winner"] = (merged["home_points"] > merged["away_points"]).astype(int)
    
    # Calculate rolling stats for this season
    merged = calculate_rolling_stats(merged)
    
    print(f"  -> Processed {len(merged)} games from {game_year} with {season_year} season stats")
    return merged

def calculate_rolling_stats(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rolling statistics for games in the dataset"""
    
    # Create long format for rolling calculations
    home_df = merged_df[["id", "season", "week", "home_team", "home_points", "away_points"]].copy()
    home_df = home_df.rename(columns={
        "home_team": "team",
        "home_points": "points_scored", 
        "away_points": "points_allowed"
    })
    home_df["home_away"] = "home"
    
    away_df = merged_df[["id", "season", "week", "away_team", "away_points", "home_points"]].copy()
    away_df = away_df.rename(columns={
        "away_team": "team",
        "away_points": "points_scored",
        "home_points": "points_allowed"
    })
    away_df["home_away"] = "away"
    
    long_df = pd.concat([home_df, away_df], ignore_index=True)
    long_df = long_df.sort_values(by=["team", "week"])
    
    # Calculate rolling stats by team
    def team_rolling_stats(group):
        group = group.sort_values('week')
        
        group["rolling_avg_points_scored"] = (
            group["points_scored"].shift(1).rolling(window=3, min_periods=1).mean()
        )
        group["rolling_avg_points_allowed"] = (
            group["points_allowed"].shift(1).rolling(window=3, min_periods=1).mean()
        )
        
        group["win"] = (group["points_scored"] > group["points_allowed"]).astype(int)
        group["rolling_win_pct"] = (
            group["win"].shift(1).rolling(window=3, min_periods=1).mean()
        )
        
        return group
    
    long_df = long_df.groupby("team").apply(team_rolling_stats).reset_index(drop=True)
    
    # Merge rolling stats back
    home_rolling = long_df[long_df["home_away"] == "home"][[
        "id", "rolling_avg_points_scored", "rolling_avg_points_allowed", "rolling_win_pct"
    ]].rename(columns={
        "rolling_avg_points_scored": "home_rolling_avg_points_scored",
        "rolling_avg_points_allowed": "home_rolling_avg_points_allowed", 
        "rolling_win_pct": "home_rolling_win_pct"
    })
    
    away_rolling = long_df[long_df["home_away"] == "away"][[
        "id", "rolling_avg_points_scored", "rolling_avg_points_allowed", "rolling_win_pct"
    ]].rename(columns={
        "rolling_avg_points_scored": "away_rolling_avg_points_scored",
        "rolling_avg_points_allowed": "away_rolling_avg_points_allowed",
        "rolling_win_pct": "away_rolling_win_pct"
    })
    
    final_merged = merged_df.merge(home_rolling, on="id", how="left")
    final_merged = final_merged.merge(away_rolling, on="id", how="left")
    
    return final_merged

def process_multiple_seasons(season_game_pairs: List[Tuple[str, str, int, int]]) -> pd.DataFrame:
    """
    Process multiple season-game file pairs and combine into one dataset
    
    Args:
        season_game_pairs: List of tuples (season_file, game_file, season_year, game_year)
        
    Returns:
        Combined DataFrame with all seasons
    """
    
    all_datasets = []
    
    for season_file, game_file, season_year, game_year in season_game_pairs:
        try:
            dataset = process_season_game_pair(season_file, game_file, season_year, game_year)
            all_datasets.append(dataset)
        except Exception as e:
            print(f"Error processing {season_file} -> {game_file}: {e}")
            continue
    
    if not all_datasets:
        raise ValueError("No datasets were successfully processed")
    
    # Combine all datasets
    print(f"\nCombining {len(all_datasets)} datasets...")
    combined_df = pd.concat(all_datasets, ignore_index=True)
    
    return combined_df

# MAIN EXECUTION
if __name__ == "__main__":
    
    # Define your season-game pairs
    # Format: (season_stats_file, games_file, season_stats_year, games_year)
    season_game_pairs = [
        ("cfb19.csv", "games2020.csv", 2019, 2020),
        ("cfb20.csv", "games2021.csv", 2020, 2021),
        ("cfb21.csv", "games2022.csv", 2021, 2022),
        ("cfb22.csv", "games2023.csv", 2022, 2023),
        ("cfb23.csv", "games2024.csv", 2023, 2024),
    ]
    
    # Process all seasons
    print("Starting multi-season processing...")
    print(f"Processing {len(season_game_pairs)} season-game pairs:")
    for season_file, game_file, season_year, game_year in season_game_pairs:
        print(f"  - {season_file} ({season_year}) -> {game_file} ({game_year})")
    
    combined_dataset = process_multiple_seasons(season_game_pairs)
    
    # Summary statistics
    print(f"\n=== FINAL DATASET SUMMARY ===")
    print(f"Total games: {len(combined_dataset)}")
    print(f"Game years: {sorted(combined_dataset['game_year'].unique())}")
    print(f"Season stat years: {sorted(combined_dataset['season_stats_year'].unique())}")
    print(f"Unique teams: {combined_dataset['home_team'].nunique()}")
    
    # Games per year
    games_per_year = combined_dataset.groupby('game_year').size()
    print(f"\nGames per year:")
    for year, count in games_per_year.items():
        print(f"  {year}: {count} games")
    
    # Check data quality
    missing_season_stats = combined_dataset.groupby('game_year')[['home_stats_missing', 'away_stats_missing']].sum()
    print(f"\nMissing season stats by year:")
    print(missing_season_stats)
    
    # Save final dataset
    output_file = "combined_multi_season_cfb_dataset.csv"
    combined_dataset.to_csv(output_file, index=False)
    print(f"\n✅ Saved combined dataset to: {output_file}")
    
    print(f"\nDataset contains:")
    print(f"- Prior season stats as baseline team features")
    print(f"- Current season game results and rolling performance")
    print(f"- Data from {combined_dataset['game_year'].min()} to {combined_dataset['game_year'].max()}")

