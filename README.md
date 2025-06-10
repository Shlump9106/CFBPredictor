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
- Rolling Values for past 5 games. How many were won, average points per game offense, average points per game defense

# How to run each group member's code
## John Heibel's Code
All of John Heibel's code is located within ./neural_net.py. To train a new model and evaluate on the test set, simply clone the repository, cd into ./neural_net and run python neural_net.py.
This will train a new model in accordance to the files and hyperparamater constants at the top of the file. A new csv file will be generated containing the results of compute_evaluation_metrics.py after every epoch. This runs on CPU and does not require GPU acceleration.


## Orion Baker's Code
All code is encompassed by the CFBDecisionTree.ipynb located in this repository. Ideally, should be run in Google Colab, and the one used for my experiments is linked here:
https://colab.research.google.com/drive/17TxEIvUzCJqZX6VKqeWncFEWpqHcZe-a?usp=sharing
Run procedurally, top to bottom, with hyperparameters tweakable at Step 5 with max_depth and random_state. You'll need to add the data files final_merged_test_2024.csv, final_merged_training_2020_2022.csv, and final_merged_valadation_2023.csv to the
colab to run it properly.

## Alen Wilson's Code
All code for my experiments is contained within the notebook logistic_regression_model.ipynb in this repository in the directory logistic_regression_py. It is best run in Google Colab, and the version I used for my experiments is linked here: 
https://colab.research.google.com/drive/1Z1vLcywoeiBn0diASFUbY2i54MjFTba9?usp=sharing
The notebook is designed to be run sequentially from top to bottom. Before training the model, I applied my entire pipeline to the standardized, merged dataset. This pipeline includes preprocessing steps such as dropping irrelevant columns, applying a custom target transformer, standardizing numerical columns, and performing feature selection. To run the notebook successfully, you’ll need to upload the following datasets to your Colab environment, final_merged_training_2020_2022.csv, final_merged_validation_2023.csv and final_merged_test_2024.csv. Additionally, all the custom transformer classes, helper functions, and utility code used in the notebook are also available in standard Python script format in the logistic_regression_py directory. To run the logistic regression model locally using these scripts, simply execute python3 logistic_regression_model.py. Note that you’ll need to have pandas, scikit-learn, and matplotlib installed in your Python environment. 
Additionaly, the standardizeDataForTest.ipynb was used to standarize our data. The Google Colab notebook is in the directory standized_data_program. You will need all the game and season data from the data directory an place it in your Google Colab enironment. To merge the data you have to adjust the season_files and game_files with the data that you want to merge. The link to this notebook is:
https://colab.research.google.com/drive/13prqGhVBPWqW-JbqHTu9STWVgFLrfNoF?usp=sharing
