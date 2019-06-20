import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from vol import Covariance as cv
import fix_yahoo_finance as yf
import pandas as pd
import os
import sys

curr_dir = sys.path[0]

class RNN(nn.Module):

    def __init__(self, data, preprocessed=False):
        super(RNN, self).__init__()

        self.num_asset = data.shape[1]

        if preprocessed:
            self.data = torch.Tensor(data)
        else:
            self.data = torch.unsqueeze(torch.Tensor(self.process_data(data)), 1)
               
        self.x_dim = self.data.shape[-1]
        self.hidden_dim = self.x_dim * 30
        self.gru = nn.GRU(self.x_dim, self.hidden_dim, 1) # one recurrent layer to discover time series relation
        self.out_layer = nn.Linear(self.hidden_dim, self.x_dim) # One output layer convert [-1, 1] to correct range
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def process_data(self, data):
        upper_data = []
        for i in range(data.shape[0]):
            cur_x = data[i,:]
            cur_x = cur_x[np.triu_indices(self.num_asset)] # Upper triangular part
            upper_data.append(cur_x.reshape([1,-1]))

        upper_data = np.concatenate(upper_data, axis=0)
        return upper_data

    def forward(self, inputs):
        seq_len, batch_size, x_dim = inputs.size()
        h0 = Variable(torch.randn(1, batch_size, self.hidden_dim))
        h_list, hn = self.gru(inputs, h0)

        output_list = []
        for i in range(seq_len):
            output_list.append(self.out_layer(h_list[i, :, :])) # convert ith hidden into ith output

        return torch.stack(output_list)

    def train(self):

        # fit in all data
        loss_list = []
        for iteration in range(1000):
            self.optimizer.zero_grad() # zero out all prev gradients
            x_pred = self.forward(self.data) 
            loss = F.mse_loss(x_pred, self.data) # use mean square loss
            loss.backward() 
            self.optimizer.step()
            if iteration%5 == 0:
                print('at iteration ', iteration, 'training loss:', loss)
            loss_list.append(loss)
        
        plt.plot(loss_list)
        plt.savefig('train_loss')
        
        x_test = self.forward(self.data)
        np.save('x_pred', torch.squeeze(x_test).detach().numpy())
        np.save('x_cov', torch.squeeze(self.data).detach().numpy())


def plot_cov():
    pred = np.load('x_pred.npy')
    real = np.load('x_cov.npy')

    for i in range(pred.shape[1]):
        cur_pred = pred[:, i]
        cur_real = real[:, i]
        plt.plot(cur_pred[20:], label='predict')
        plt.plot(cur_real[20:], label='real')
        plt.savefig('cov'+str(i))
        plt.close()



if __name__ == '__main__':
    load = True
    # if load == False:
        # data = pd.read_csv('yahoo_data.csv')
    
    data = yf.download(['SPY','MCHI','QQQ','ICLN','QCLN','IBB','XBI','FBT','IVV','IYR','SCHH'],start='2007-01-01',end='2019-04-22')
    data.to_csv('yahoo_data.csv')

    adjclose = data['Adj Close']
    adjclose1 = adjclose.dropna()
    selected = ['SPY','QQQ','IBB','XBI','FBT','IYR','SCHH']
    portfolio = adjclose1[adjclose1.columns[adjclose1.columns.isin(selected)]]
    timeoption = None
    info = cv.find_cov(portfolio,timeoption)   
    #obtain the covaraince
    cov = info[0]
    cov = np.array(cov)

    # Fit into the rnn model
    rnn_model = RNN(cov)
    rnn_model.train()
   
    plot_cov()

