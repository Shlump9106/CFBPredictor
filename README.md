# CFBPredictor
College football game predictor based on seasonal and game data. Uses decision tree, logistic regression, and feedforward neural networks.


# Data Processing:
- Two separate datasets:
- Season Data (rich, many features)
- https://www.kaggle.com/datasets/jeffgallini/college-football-team-stats-2019?resource=download&select=cfb23.csv
- Game Data (game-by-game, few features, but includes the data we are trying to predict as example)
- https://github.com/jackwarfield/cfb_rankings/blob/main/games2020.csv

Each record in the dataset is the winning teams points earned in the past historical season. We can add rolling values from previous games this season.
We will make a dataloader.py file that will use these two datasets, merging them so that one singular set of data is used for the entirety of the project.
All three models (Orion: Decision Tree, Alen: Logistic Regression, John: Neural Network) will train on the exact same data so performance is based on a standardized starting point.
A single line item for learning would consist of the following:
- Week, Neutral Site, Conference Game, Home Team ID, Home Team Points, Away Team ID, Away Team Points, Home Conference Level, Away Conference Level, **Home Team Win/Lose** (from Game Data)
- Entirety of Season Data from the past year's season as individual columns, for each team (~300 columns)
- Rolling Values for past n games. How many were won, average points per game offense, average points per game defense
