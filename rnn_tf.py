import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import tensorflow as tf
from vol import Covariance as cv
import fix_yahoo_finance as yf
import pandas as pd

def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.), dtype=tf.float32)

class RNN(object):
  def __init__(self, data, batch_size=22):
    self.num_asset = data.shape[1]
    self.batch_size = batch_size
    self.x_dim = int(self.num_asset*(self.num_asset+1)/2)
    self.hidden_dim = self.x_dim * 20
    self.Wxh = tf.Variable(glorot_init([self.x_dim , self.hidden_dim]))
    self.Whh = tf.Variable(glorot_init([self.hidden_dim, self.hidden_dim]))
    self.Why = tf.Variable(glorot_init([self.hidden_dim, self.x_dim ]))
    self.bh = tf.Variable(tf.zeros([self.hidden_dim]))
    self.by = tf.Variable(tf.zeros([self.x_dim]))

    upper_data = []
    for i in range(data.shape[0]):
        cur_x = data[i,:]
        cur_x = cur_x[np.triu_indices(self.num_asset)] # Upper triangular part
        upper_data.append(cur_x.reshape([1,-1]))
        # print("####cur_x", cur_x)

    upper_data = np.concatenate(upper_data, axis=0)
    # print('#######upper_data:', upper_data, 'upper_data shape:', upper_data.shape)
    self.data = upper_data
   

  def run(self):
    self.train()
    pred = self.model(self.data[-22, :])
    return pred

  # Build RNN Graph
  def model(self, X): 
    X = tf.log(X)
    # rolling_wind = X.shape[0]
    # print(rolling_wind)
    pre_h = tf.zeros([1, self.hidden_dim]) 
    y_list = []
    for i in range(self.batch_size):
        cur_x = X[i,:]
        cur_x = tf.reshape(cur_x, [1, -1])
        h = tf.matmul(cur_x, self.Wxh) + tf.matmul(pre_h, self.Whh) + self.bh
        pre_h = h
        cur_y = tf.matmul(h, self.Why) + self.by
        y_list.append(cur_y)

    y_list = tf.concat(y_list, 0)

    return y_list

  def train(self):
    inputs = tf.placeholder(tf.float32, [None, self.x_dim], name='inputs')
    y = self.model(inputs)
    loss = tf.losses.mean_squared_error(labels=inputs, predictions=y)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    grads = tf.gradients(loss, [self.Wxh, self.Whh, self.Why, self.bh, self.by])
    data_len = int(self.data.shape[0])

    init_vars = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_vars)
        for epoch in range(1):
            for i in range(data_len-self.batch_size):
                batch = self.data[i:i+self.batch_size,:]
                optimizer.run(feed_dict={inputs:batch})
                # if i%5 ==0:
                pred, train_loss, weight_grad = sess.run([y, loss, grads], feed_dict={inputs:batch})
                print('gradients: ', weight_grad)
                print('x:', batch, 'y:', pred)
                print('at iter %d, train loss is %f'%(i, train_loss))

        # test_loss = loss.eval(feed_dict={inputs:self.data})
        # print('test loss is', test_loss)


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
    # print('cov:', cov, 'dic:', dic)
    rnn_model = RNN(cov, len(selected))
    pred = rnn_model.run()


