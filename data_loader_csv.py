from __future__ import annotations

"""
DataLoader
==========

A unified loader that merges season-level statistics with game-level logs for
college-football modelling tasks.

…  (rest of your docstring unchanged)  …
"""

from pathlib import Path
from typing import Iterable, Dict, Tuple
import pandas as pd
import re

__all__ = ["DataLoader"]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _clean_col(name: str) -> str:
    """
    Convert a raw column header to lower-snake_case and replace any
    whitespace, dots or dashes with a single underscore.
    """
    return re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower())


# ---------------------------------------------------------------------------
# Team name mapping helper
# ---------------------------------------------------------------------------
class TeamNameMapper:
    """Translate the different spellings used by the two data sources."""

    REQUIRED_COLS = {"games", "season", "canonical"}

    def __init__(self, mapping_csv: str | Path):
        path = Path(mapping_csv)
        if not path.exists():
            raise FileNotFoundError(path)
        self._games_to_canon, self._season_to_canon = self._load_mapping(path)

    def _load_mapping(self, path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
        df = pd.read_csv(path)
        missing = self.REQUIRED_COLS.difference(df.columns)
        if missing:
            raise ValueError(
                f"Mapping file must contain columns {sorted(self.REQUIRED_COLS)}, "
                f"but is missing {sorted(missing)}"
            )
        # Strip whitespace in all object columns
        df = df.apply(lambda s: s.str.strip() if s.dtype == "object" else s)
        games_to_canon: Dict[str, str] = dict(zip(df["games"], df["canonical"]))
        season_to_canon: Dict[str, str] = dict(zip(df["season"], df["canonical"]))
        return games_to_canon, season_to_canon

    def games_name_to_canonical(self, name: str) -> str | None:
        """
        Return the canonical name for a 'games' entry.
        If no mapping exists, return None.
        """
        return self._games_to_canon.get(name)

    def season_name_to_canonical(self, name: str) -> str | None:
        """
        Return the canonical name for a 'season' entry.
        If no mapping exists, return None.
        """
        return self._season_to_canon.get(name)


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
class DataLoader:
    """Merge season and game datasets into a modelling table."""

    def __init__(
        self,
        season_files:   Iterable[str | Path],
        game_files:     Iterable[str | Path],
        mapping_csv:    str | Path,
        rolling_window: int = 5,
    ) -> None:
        self.season_files   = [Path(f) for f in season_files]
        self.game_files     = [Path(f) for f in game_files]
        self.mapper         = TeamNameMapper(mapping_csv)
        self.rolling_window = int(rolling_window)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def build(self) -> pd.DataFrame:
        """Return the fully merged feature table."""
        season_df = self._load_season_data()
        games_df  = self._load_game_data()

        merged = self._merge_games_with_season(season_df, games_df)
        merged = self._add_rolling_features(merged)
        return merged

    def train_valid_test_split(
        self,
        df: pd.DataFrame | None = None,
        *,
        train_years: Iterable[int],
        valid_years: Iterable[int],
        test_years:  Iterable[int],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Slice a pre-built table into (train, valid, test) by game season."""
        if df is None:
            df = self.build()
        train = df[df["season"].isin(train_years)].reset_index(drop=True)
        valid = df[df["season"].isin(valid_years)].reset_index(drop=True)
        test  = df[df["season"].isin(test_years)].reset_index(drop=True)
        return train, valid, test

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_season_data(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for f in self.season_files:
            df = pd.read_csv(f)
            # Drop extraneous index columns written by some CSV exporters
            df = df.loc[:, ~df.columns.str.contains(r"^unnamed", case=False)]
            # Normalise column headers
            df.columns = [_clean_col(c) for c in df.columns]
            if "team" not in df.columns:
                raise ValueError(f"Expected a column 'Team' (any case) in {f}")

            # Canonicalise team names, dropping those not in mapping
            df["team"] = df["team"].apply(self.mapper.season_name_to_canonical)
            df = df[df["team"].notna()]

            # Add season year if not present
            if "season" not in df.columns:
                # infer from file name e.g. cfb18.csv → 2018
                year = int("20" + f.stem[-2:])
                df.insert(0, "season", year)

            # Parse "win_loss" into separate numeric columns if present
            if "win_loss" in df.columns:
                wl = df["win_loss"].str.split("-", expand=True)
                if wl.shape[1] == 2:
                    df["wins"]   = pd.to_numeric(wl[0], errors="coerce").astype("Int64")
                    df["losses"] = pd.to_numeric(wl[1], errors="coerce").astype("Int64")
                df.drop(columns=["win_loss"], inplace=True)
            elif "win" in df.columns and "loss" in df.columns:
            # Rename to consistent columns 'wins' and 'losses'
              df["wins"] = pd.to_numeric(df["win"], errors="coerce").astype("Int64")
              df["losses"] = pd.to_numeric(df["loss"], errors="coerce").astype("Int64")
              df.drop(columns=["win", "loss"], inplace=True)

            # Robust numeric coercion (strings → numbers, keep NaN for non-numeric)
            for col in df.columns.difference(["team", "season"]):
                if df[col].dtype == "object":
                    num = pd.to_numeric(
                        df[col].astype(str).str.replace(",", ""),
                        errors="coerce"
                    )
                    # Promote to nullable Int64 if every non-NaN value has no fractional part
                    non_na = num.dropna()
                    if non_na.empty or ((non_na % 1) == 0).all():
                        df[col] = num.astype("Int64")
                    else:
                        df[col] = num

            # ----------------------------------------------------------------
            # ** NEW: Drop the two season-level columns that were always empty **
            # (so they never become home_/away_time_of_possession, etc.)
            df.drop(
                columns=[
                    "time_of_possession",
                    "average_time_of_possession_per_game",
                ],
                inplace=True,
                errors="ignore"
            )
            # ----------------------------------------------------------------

            frames.append(df)
        season_df = pd.concat(frames, ignore_index=True)
        return season_df

    def _load_game_data(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for f in self.game_files:
            df = pd.read_csv(f)
            # Normalise column headers
            df.columns = [_clean_col(c) for c in df.columns]

            # Basic dtype coercions
            bool_cols = ["neutral_site", "conference_game"]
            for bc in bool_cols:
                if bc in df.columns:
                    df[bc] = df[bc].astype(bool)

            num_cols = [
                "id",
                "season",
                "week",
                "home_id",
                "home_points",
                "away_id",
                "away_points",
            ]
            for nc in num_cols:
                if nc in df.columns:
                    num = pd.to_numeric(df[nc], errors="coerce")
                    non_na = num.dropna()
                    if non_na.empty or ((non_na % 1) == 0).all():
                        df[nc] = num.astype("Int64")
                    else:
                        df[nc] = num

            if "start_date" in df.columns:
                df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")

            # Canonicalise team names, dropping any games where a team isn't in mapping
            df["home_team"] = df["home_team"].apply(self.mapper.games_name_to_canonical)
            df["away_team"] = df["away_team"].apply(self.mapper.games_name_to_canonical)
            df = df[df["home_team"].notna() & df["away_team"].notna()]

            # Determine winner column (requires home_points and away_points)
            df["winner"] = df.apply(
                lambda r: "home" if r["home_points"] > r["away_points"] else "away",
                axis=1
            )

            df["winner"] = df["winner"].map({"home": 1, "away": 0})

            frames.append(df)
        games_df = pd.concat(frames, ignore_index=True)
        return games_df

    def _merge_games_with_season(self, season_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
        """Attach season-level stats to both home and away teams for each game."""
        # We'll perform two separate merges, then prefix columns
        home_season = season_df.add_prefix("home_")
        away_season = season_df.add_prefix("away_")

        # Merge on season & canonical team name
        merged = (
            games_df
            .merge(
                home_season,
                left_on=["season", "home_team"],
                right_on=["home_season", "home_team"],
                how="inner"  # only keep games where season info exists
            )
            .merge(
                away_season,
                left_on=["season", "away_team"],
                right_on=["away_season", "away_team"],
                how="inner"  # only keep games where season info exists
            )
        )
        # Clean up duplicate key cols from right frames
        dup_cols = [c for c in merged.columns if c.endswith("_season")]
        merged.drop(columns=dup_cols, inplace=True)
        # Remove other automatic `_x` / `_y` duplicates from the join
        dup_key_cols = [c for c in merged.columns if c.endswith(("_x", "_y"))]
        merged.drop(columns=dup_key_cols, inplace=True, errors="ignore")
        return merged

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling wins, OFF PPG, DEF PPG over the past *n* games."""
        window = self.rolling_window
        games = (
            df[["season", "week", "home_team", "away_team", "home_points", "away_points", "winner"]]
            .copy()
        )

        # Reshape: one row per team per game (long format)
        long_home = games.rename(columns={
            "home_team": "team",
            "home_points": "team_points",
            "away_points": "opp_points",
        })[["season", "week", "team", "team_points", "opp_points", "winner"]]
        long_home["is_win"] = (long_home["winner"] == "home").astype(int)

        long_away = games.rename(columns={
            "away_team": "team",
            "away_points": "team_points",
            "home_points": "opp_points",
        })[["season", "week", "team", "team_points", "opp_points", "winner"]]
        long_away["is_win"] = (long_away["winner"] == "away").astype(int)

        long = pd.concat([long_home, long_away], ignore_index=True)
        long.sort_values(["team", "season", "week"], inplace=True)

        # Rolling calculations (exclude current game with shift)
        grouped = long.groupby(["team", "season"], group_keys=False)
        long["rolling_wins"] = grouped["is_win"].shift().rolling(window, min_periods=1).sum()
        long["rolling_off_ppg"] = grouped["team_points"].shift().rolling(window, min_periods=1).mean()
        long["rolling_def_ppg"] = grouped["opp_points"].shift().rolling(window, min_periods=1).mean()

        # Merge back
        long_features = long[["season", "week", "team", "rolling_wins", "rolling_off_ppg", "rolling_def_ppg"]]
        df = (
            df.merge(
                long_features.add_prefix("home_"),
                left_on=["season", "week", "home_team"],
                right_on=["home_season", "home_week", "home_team"],
                how="left"
            )
            .merge(
                long_features.add_prefix("away_"),
                left_on=["season", "week", "away_team"],
                right_on=["away_season", "away_week", "away_team"],
                how="left"
            )
        )
        df.drop(columns=[c for c in df.columns if c.endswith(("_season", "_week"))], inplace=True)
        return df


# ---------------------------------------------------------------------------
# Script entry-point (modified to always write CSV)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json, sys

    parser = argparse.ArgumentParser(
        description="Merge season and game datasets for CFB modelling."
    )
    parser.add_argument("--seasons", nargs="+", help="Season CSVs (cfbYY.csv)")
    parser.add_argument("--games",   nargs="+", help="Game CSVs   (gamesYYYY.csv)")
    parser.add_argument(
        "--mapping", required=True,
        help="CSV mapping file with columns games,season,canonical"
    )
    parser.add_argument(
        "--rolling", type=int, default=5,
        help="Rolling window size (games)"
    )
    parser.add_argument(
        "--out", required=True,
        help="Output path (will be written as CSV, regardless of extension)"
    )
    args = parser.parse_args()

    dl = DataLoader(args.seasons, args.games, args.mapping, args.rolling)
    df = dl.build()
    df = df.drop_duplicates(subset=["id"])

    # Force CSV output, ignoring whatever extension was passed in.
    out_path = Path(args.out)
    df.to_csv(out_path, index=False)

    # Print summary
    print(json.dumps({"rows": len(df), "columns": len(df.columns)}, indent=2))