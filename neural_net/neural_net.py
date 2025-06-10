"""Train a feedforward neural network for binary home team win prediction using PyTorch."""

import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from evaluation import compute_evaluation_metrics, print_evaluation_results
from load_and_process import load_and_preprocess, CFBDataset
import csv

# constants
TRAIN_CSV = '../final_merged_training_2020_2022.csv'
TEST_CSV = '../final_merged_test.csv'
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 5e-4
HIDDEN_DIM = 96
EMBED_DIM = 48
WEIGHT_DECAY = 1e-4
STEP_SIZE = 5
GAMMA = 0.5
SEED = 3

class Net(nn.Module):
    def __init__(self, num_feats, num_teams, embed_dim, hidden_dim):
        super().__init__()
        # Embed the teams rather than one hot encode since we have a large nuber of them, and we want the model to learn their relative abilities
        self.team_embed = nn.Embedding(num_teams, embed_dim)
        self.fc = nn.Sequential(
            # Small neural net with a single hidden layer
            nn.Linear(num_feats + 2 * embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, home, away):
        #forward pass where we get the logits of numeric features + home and away embeddings
        h_emb = self.team_embed(home)
        a_emb = self.team_embed(away)
        x = torch.cat([x, h_emb, a_emb], dim=1)
        return self.fc(x).squeeze(-1)

def main():
    import random
    # set seed for testing
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # load features from our training csv
    (X_train, home_train, away_train, y_train, X_test, home_test, away_test, y_test, feature_names, team_mapping) = load_and_preprocess(TRAIN_CSV, TEST_CSV)
    print(f"Number of features in training dataset: {len(feature_names)}")
    # put data into CFBDataset objects
    train_dataset = CFBDataset(X_train, home_train, away_train, y_train)
    test_dataset = CFBDataset(X_test, home_test, away_test, y_test)
    # use pytorch loade
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # load to cpu, given the amount of data we have, realistically we don't need gpu acceleration
    device = torch.device('cpu')
    model = Net(len(feature_names), len(team_mapping), EMBED_DIM, HIDDEN_DIM).to(device)
    # use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # use learning rate scheduler
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)


    #use sigmoind binary cross entropy loss as our loss function
    loss_function = nn.BCEWithLogitsLoss()

    # store results for graphing for report
    csv_file = 'training_results.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'learning_rate', 'accuracy', 'balanced_accuracy', 'f1_score', 'precision', 'recall', 'roc_auc'])

    # main training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for features_batch, home_ids_batch, away_ids_batch, labels_batch in train_loader:
            # send batches to cpu
            features_batch = features_batch.to(device)
            home_ids_batch = home_ids_batch.to(device)
            away_ids_batch = away_ids_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            predictions = model(features_batch, home_ids_batch, away_ids_batch)
            loss = loss_function(predictions, labels_batch)

            # calculate loss through backprop
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * features_batch.size(0)

        scheduler.step()

        average_loss = epoch_loss / len(train_dataset)
        current_learning_rate = optimizer.param_groups[0]['lr']
        # print current epoch stats
        print(f'Epoch {epoch}/{EPOCHS} loss={average_loss:.4f} lr={current_learning_rate:.2e}')

        #during training eval
        model.eval()
        all_test_probs = []

        with torch.no_grad():
            for features_batch, home_ids_batch, away_ids_batch, labels_batch in test_loader:
                features_batch = features_batch.to(device)
                home_ids_batch = home_ids_batch.to(device)
                away_ids_batch = away_ids_batch.to(device)

                logits = model(features_batch, home_ids_batch, away_ids_batch)
                probabilities = torch.sigmoid(logits)
                all_test_probs.extend(probabilities.cpu().numpy().tolist())
        # get eval metrics during training
        test_results = compute_evaluation_metrics(y_test, np.array(all_test_probs))
        print_evaluation_results(test_results)

        # log vals
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, average_loss, current_learning_rate, test_results['accuracy'], test_results['balanced_accuracy'], test_results['f1_score'], test_results['precision'], test_results['recall'], test_results['roc_auc']])


    baseline_accuracy = np.mean(y_test == 1)
    print(f'Baseline accuracy = {baseline_accuracy:.4f}')

    model.eval()
    all_probabilities = []
    all_predictions = []

    with torch.no_grad():
        for features_batch, home_ids_batch, away_ids_batch, labels_batch in test_loader:
            features_batch = features_batch.to(device)
            home_ids_batch = home_ids_batch.to(device)
            away_ids_batch = away_ids_batch.to(device)

            logits = model(features_batch, home_ids_batch, away_ids_batch)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()

            all_probabilities.extend(probabilities.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    evaluation_results = compute_evaluation_metrics(y_test, np.array(all_probabilities))
    print_evaluation_results(evaluation_results)


if __name__ == '__main__':
    main()