import numpy as np
import math
from matplotlib import pyplot as plt
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys


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


def Feedforward(input_size, hidden_size, num_layers=2, p=[0.05]):

    layer_list = []
    in_features = input_size
    for i in range(num_layers):
        out_features = hidden_size[i]
        layer_list.append(nn.Linear(in_features, out_features))
        layer_list.append(nn.ReLU())
        layer_list.append(nn.Dropout(p[i]))
        in_features = out_features
    layer_list.append(nn.Linear(in_features, 1))
    layers = nn.Sequential(*layer_list)

    return layers


if __name__ == '__main__':
    print('Getting data')
    obs = load_np_data('hourly_data_2020-2021.npy')  # load observations
    X = load_np_data('predictors_2020-2021.npy')  # load input

    # normalize
    min_obs = np.min(obs)
    max_obs = np.max(obs)
    obs = (obs - min_obs) / (max_obs-min_obs)
    for i in range(X.shape[0]):
        X[i,:] = (X[i,:] - np.min(X[i,:])) / (np.max(X[i,:]) - np.min(X[i,:]))  # normalize

    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(X, obs)  # split into test + train

    print('Training network')
    model = Feedforward(X_train.shape[1],[89,88],num_layers=2,p=[0.05,0])
    optimizer = optim.Adam(model.parameters(), lr=0.00012104)
    criterion = nn.MSELoss()

    model.eval()
    y_pred = model(X_test)
    before_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss before training' , before_train.item())

    epoch = 600
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

            # if epoch % 20 == 0 and i == 49:
            #     plt.plot(range(len(y_batch_train)), y_pred.detach().squeeze(),label='model')
            #     plt.plot(range(len(y_batch_train)), y_batch_train.detach().squeeze(),label='test batch')
            #     plt.legend()
            #     plt.title('Test batch at this epoch')
            #     plt.show()

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
            if epoch > 150:
                print('New minimum:',valid_loss.item(), ' at epoch ', epoch)
        if epoch % 20 == 0:
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss,valid_loss))
        # if epoch % 20 == 0:
        #     plt.plot(range(100), y_pred.detach().squeeze()[:100],label='model')
        #     plt.plot(range(100), y_valid[:100],label='validation')
        #     plt.legend()
        #     plt.title('Validation batch at this epoch')
        #     plt.show()

    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    y_pred = model(X_test)
    after_train = criterion(y_pred.squeeze(), y_test) 
    print('Test loss after Training' , after_train.item())


    plt.plot(range(50),y_test[:50],label='test')
    plt.plot(range(50), y_pred.detach().squeeze()[:50],label='model')
    plt.ylabel('Normalized wind speed')
    plt.legend()
    plt.show()

    print('Done')

