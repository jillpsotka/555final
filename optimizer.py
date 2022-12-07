import numpy as np
import math
from matplotlib import pyplot as plt
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import optuna
from optuna.trial import TrialState


def load_np_data(file):
    # load data np array
    data = np.load(file)
    return data


def split_data(x, y, p_train=0.9, p_valid=0.9):
    # split all data into testing and training
    # shuffle before split to avoid bias
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    x = x[:,idx]
    y = y[idx]

    ntrain = round(p_train*len(y))
    nvalid = round(p_valid*ntrain)

    x_train = x[:,:nvalid]
    x_train = torch.FloatTensor(x_train.T)
    y_train = torch.FloatTensor(y[:nvalid])

    x_valid = x[:,nvalid:ntrain]
    x_valid = torch.FloatTensor(x_valid.T)
    y_valid = torch.FloatTensor(y[nvalid:ntrain])    

    x_test = x[:,ntrain:]
    x_test = torch.FloatTensor(x_test.T)
    y_test = torch.FloatTensor(y[ntrain:])

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def define_model(input_size, trial):
    num_layers = trial.suggest_int('n_layers', 1, 8)
    
    layer_list = []
    in_features = input_size
    for i in range(num_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 20, 90)
        layer_list.append(nn.Linear(in_features, out_features))
        layer_list.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0, 0.5, step=0.05)
        layer_list.append(nn.Dropout(p))
        in_features = out_features
    layer_list.append(nn.Linear(in_features, 1))
    layers = nn.Sequential(*layer_list)

    return layers



def objective(trial):
    # for optuna to optimize hyperparameters

    lr = trial.suggest_float('lr',1e-5, 1e-2, log=True)

    model = define_model(X_train.shape[1],trial)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epoch = 300
    batch_size = 4
    num_batch = round(0.81*8000/batch_size)  # ~8,000 data points per year
    idx = np.arange(X_train.shape[0])
    valid_loss_min = np.Inf
    for epoch in range(epoch):
        model.train()
        train_loss = 0

        # Shuffle data every epoch
        np.random.shuffle(idx)
        X_shuffled = X_train[idx,:]
        y_shuffled = y_train[idx]
        batches = np.array_split(X_shuffled, num_batch, axis=0)
        y_batches = np.array_split(y_shuffled, num_batch)

        for i in range(num_batch):
            batch_train = batches[i]
            y_batch_train = y_batches[i]
            optimizer.zero_grad()
            # Forward pass
            y_pred = model(batch_train)
            # Compute Loss
            loss = criterion(y_pred.squeeze(), y_batch_train)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()  # validation
        y_pred = model(X_valid)
        valid_loss = criterion(y_pred.squeeze(),y_valid)
        train_loss = train_loss / num_batch
        if valid_loss <= valid_loss_min:
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss

        trial.report(valid_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    y_pred = model(X_test)
    after_train = criterion(y_pred.squeeze(), y_test)
    return after_train



if __name__ == '__main__':
    print('Getting data')
    obs = load_np_data('hourly_data_2021.npy')  # load observations
    X = load_np_data('predictors_2021.npy')  # load input

    # normalize
    min_obs = np.min(obs)
    max_obs = np.max(obs)
    obs = (obs - min_obs) / (max_obs-min_obs)
    for i in range(X.shape[0]):
        X[i,:] = (X[i,:] - np.min(X[i,:])) / (np.max(X[i,:]) - np.min(X[i,:]))  # normalize

    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(X, obs)  # split into test + train

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

