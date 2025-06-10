import pandas as pd
import re
import torch
from torch.utils.data import DataLoader, Dataset


def normalize_column_name(column_name):
    """Deal with non standard chars in our col names"""
    s = column_name.strip().lower()
    chars = []
    for c in s:
        if c.isalnum():
            chars.append(c)
        else:
            chars.append('_')
    result_chars = []
    prev_char = None
    for c in chars:
        if c == '_' and prev_char == '_':
            continue
        result_chars.append(c)
        prev_char = c
    normalized = ''.join(result_chars).strip('_')
    return normalized


def load_dataframe(data_input):
    """load from csv"""
    return pd.read_csv(data_input)


def normalize_column_names(dataframe):
    """normalize all col names with earlier function"""
    normalized_columns = []
    for column in dataframe.columns:
        normalized_columns.append(normalize_column_name(column))
    dataframe.columns = normalized_columns


def rename_test_columns(test_df):
    """test data has some different names, so we have to rename"""
    column_mapping = {
        'home_rolling_avg_points_scored': 'home_rolling_off_ppg',
        'home_rolling_avg_points_allowed': 'home_rolling_def_ppg',
        'home_rolling_win_pct': 'home_rolling_wins',
        'away_rolling_avg_points_scored': 'away_rolling_off_ppg',
        'away_rolling_avg_points_allowed': 'away_rolling_def_ppg',
        'away_rolling_win_pct': 'away_rolling_wins',
    }
    test_df.rename(columns=column_mapping, inplace=True)


def standardize_column_prefixes(test_df):
    """Add home and away prefix to sgame"""
    id_columns = [
        'id', 'season', 'week', 'season_type', 'start_date',
        'neutral_site', 'conference_game', 'home_id', 'away_id',
        'home_team', 'away_team', 'home_points', 'away_points',
        'home_level', 'away_level', 'winner'
    ]
    
    new_columns = []
    for column in test_df.columns:
        if column in id_columns or column.startswith('home_') or column.startswith('away_'):
            new_columns.append(column)
        elif column.endswith('_away'):
            base_name = column[:-len('_away')]
            new_columns.append(f"away_{base_name}")
        else:
            new_columns.append(f"home_{column}")
    
    test_df.columns = new_columns


def get_columns_to_drop():
    """Return list of columns to drop from both dataframes."""
    return [
        'home_points', 'away_points', # Can't include because then model just learns to predict outcome on points
        'home_team', # Not really helpful to train on strings since we have team names
        'away_team',
         'season_type', 'start_date' #Not really helpful either
    ]


def find_unnamed_columns(dataframe):
    """get unnamed cols"""
    unnamed_columns = []
    for column in dataframe.columns:
        if 'unnamed' in column.lower():
            unnamed_columns.append(column)
    return unnamed_columns


def clean_dataframes(train_df, test_df):
    """clean dataset of unnamed cols and turn bools to ints"""
    columns_to_drop = get_columns_to_drop()
    
    # Add unnamed columns to drop list
    train_unnamed = find_unnamed_columns(train_df)
    test_unnamed = find_unnamed_columns(test_df)
    columns_to_drop.extend(train_unnamed)
    columns_to_drop.extend(test_unnamed)
    
    # convert bool neutral and conference into int
    for dataframe in [train_df, test_df]:
        for column in ['neutral_site', 'conference_game']:
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].astype(int)
        
        # drop all marked columns, both unnamed and
        existing_drop_columns = []
        for column in columns_to_drop:
            if column in dataframe.columns:
                existing_drop_columns.append(column)
        dataframe.drop(columns=existing_drop_columns, inplace=True)


def extract_targets(train_df, test_df):
    """extract target vars we are predicting"""
    y_train = train_df['winner'].values
    y_test = test_df['winner'].values
    train_df.drop(columns=['winner'], inplace=True)
    test_df.drop(columns=['winner'], inplace=True)
    return y_train, y_test


def get_common_numeric_features(train_df, test_df):
    """Find numeric cols that exist in both train and test"""
    categorical_columns = ['home_id', 'away_id']
    
    # get numeric columns that we don't need to embed / one hot encode
    train_numeric_columns = []
    for column in train_df.columns:
        if pd.api.types.is_numeric_dtype(train_df[column]) and column not in categorical_columns:
            train_numeric_columns.append(column)
    
    # make sure those features are in common with test dataset, as there have been some issues
    common_features = []
    for column in train_numeric_columns:
        if column in test_df.columns:
            common_features.append(column)
    
    return common_features


def create_difference_features(train_df, test_df, common_features):
    """Create difference features between home and away team stats. Still a large number of features but much more manageable"""
    # Find home/away feature pairs
    feature_pairs = []
    for feature in common_features:
        if feature.startswith('home_'):
            base_feature = feature[len('home_'):]
            away_feature = f'away_{base_feature}'
            if away_feature in common_features:
                feature_pairs.append((feature, away_feature))

    # Create difference features
    difference_features = []
    train_diff_data = {}
    test_diff_data = {}

    for home_feature, away_feature in feature_pairs:
        base_name = home_feature[len('home_'):]
        diff_name = f"diff_{base_name}"
        train_diff_data[diff_name] = train_df[home_feature] - train_df[away_feature]
        test_diff_data[diff_name] = test_df[home_feature] - test_df[away_feature]
        difference_features.append(diff_name)

    # Add difference features to dataframes
    if train_diff_data:
        train_diff_df = pd.DataFrame(train_diff_data, index=train_df.index)
        test_diff_df = pd.DataFrame(test_diff_data, index=test_df.index)
        train_df = pd.concat([train_df, train_diff_df], axis=1)
        test_df = pd.concat([test_df, test_diff_df], axis=1)

    # remove original features that were used to create differences
    # use sets so its easy to add and we can confirm no duplicates
    features_to_remove = set()
    for home_feature, away_feature in feature_pairs:
        features_to_remove.add(home_feature)
        features_to_remove.add(away_feature)

    # update feature list
    final_features = []
    for feature in common_features:
        if feature not in features_to_remove:
            final_features.append(feature)
    final_features.extend(difference_features)

    #print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}, final features: {final_features}")
    return train_df, test_df, final_features


def normalize_features(X_train, X_test):
    """norm features using training set statistics."""
    feature_means = X_train.mean()
    feature_stds = X_train.std().replace(0, 1)
    
    X_train_normalized = (X_train - feature_means) / feature_stds
    # also norm test based off of train average
    X_test_normalized = (X_test - feature_means) / feature_stds

    # get rid of nans. Theres a good number of them in the seasonal dataset
    X_train_normalized = X_train_normalized.fillna(0)
    X_test_normalized = X_test_normalized.fillna(0)
    
    return X_train_normalized, X_test_normalized


def create_team_id_mapping(train_df, test_df):
    """create team ids into dict"""
    all_team_ids = set()
    all_team_ids.update(train_df['home_id'])
    all_team_ids.update(train_df['away_id'])
    all_team_ids.update(test_df['home_id'])
    all_team_ids.update(test_df['away_id'])
    
    sorted_team_ids = sorted(all_team_ids)
    
    id_mapping = {}
    for index, team_id in enumerate(sorted_team_ids):
        id_mapping[team_id] = index
    
    return id_mapping


def map_team_ids(train_df, test_df, id_mapping):
    """Get team ids from mapping"""
    home_train_ids = train_df['home_id'].map(id_mapping).values
    away_train_ids = train_df['away_id'].map(id_mapping).values
    home_test_ids = test_df['home_id'].map(id_mapping).values
    away_test_ids = test_df['away_id'].map(id_mapping).values
    
    return home_train_ids, away_train_ids, home_test_ids, away_test_ids


def load_and_preprocess(train_csv, test_csv):
    """
    use the previous functions to actually make our dataset vals for train and test at the same time
    """
    # load data
    train_df = load_dataframe(train_csv)
    test_df = load_dataframe(test_csv)
    
    # norm cols
    normalize_column_names(train_df)
    normalize_column_names(test_df)
    
    # rename and standardize
    rename_test_columns(test_df)
    standardize_column_prefixes(test_df)
    
    # Clean
    clean_dataframes(train_df, test_df)
    
    # Extract targets
    y_train, y_test = extract_targets(train_df, test_df)
    
    # Get common num features
    common_features = get_common_numeric_features(train_df, test_df)
    
    # Make sure we are actually training off of common features
    train_df, test_df, final_features = create_difference_features(train_df, test_df, common_features)
    
    # get features
    X_train = train_df[final_features].astype(float).copy()
    X_test = test_df[final_features].astype(float).copy()
    
    # norm features
    X_train, X_test = normalize_features(X_train, X_test)
    
    # create team id mapping
    team_id_mapping = create_team_id_mapping(train_df, test_df)
    
    # Map team ids
    home_train_ids, away_train_ids, home_test_ids, away_test_ids = map_team_ids(train_df, test_df, team_id_mapping)

    # get values
    return (X_train.values, home_train_ids, away_train_ids, y_train, X_test.values, home_test_ids, away_test_ids, y_test, final_features, team_id_mapping)


class CFBDataset(Dataset):
    """dataset object for train and test"""
    
    def __init__(self, X, home_ids, away_ids, targets):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.home_ids = torch.tensor(home_ids, dtype=torch.long)
        self.away_ids = torch.tensor(away_ids, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        """overload used for length"""
        return len(self.targets)

    def __getitem__(self, idx):
        """overload used to grab sample"""
        return self.X[idx], self.home_ids[idx], self.away_ids[idx], self.targets[idx]