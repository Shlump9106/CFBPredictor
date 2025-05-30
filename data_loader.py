# =============================================================================
# DataLoader Module
#
# This module merges and prepares two datasets for model training:
#
# 1. Season Data:
#    - Rich, feature-heavy historical season statistics
#    - Source: https://www.kaggle.com/datasets/jeffgallini/college-football-team-stats-2019
#
# 2. Game Data:
#    - Game-by-game results (includes target variable: points earned by winning team)
#    - Source: https://github.com/jackwarfield/cfb_rankings/blob/main/games2020.csv
#
# Each record represents a winning team’s points from a past season. We will:
#   1. Merge season data (≈300 features per team) with game-level data.
#   2. Compute rolling statistics over the past n games:
#      - Number of wins
#      - Average offensive points per game
#      - Average defensive points allowed per game
#
# Final feature set for each training example:
#   - Week
#   - Neutral Site (bool)
#   - Conference Game (bool)
#   - Home Team ID
#   - Home Team Points
#   - Away Team ID
#   - Away Team Points
#   - Home Conference Level
#   - Away Conference Level
#   - Home Team Win/Loss (target)
#   - Season Data features (past season)
#   - Rolling game-by-game features (past n games)
# =============================================================================
