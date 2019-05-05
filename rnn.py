import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import tensorflow as tf

def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.), dtype=tf.float32)

class rnn(object):
  def __init__(self, data, num_asset=20, rolling_window=22):
  	# Reshape the data
  	# Only Keep the upper traingle of Data
  	upper_data = []
	for i in range(data.shape[0]):
		cur_x = data[i,:]
		cur_x = tf.matrix_band_part(cur_x, 0, -1) # Upper triangular part
    	cur_x = tf.reshape(cur_x, [-1]) #Flatten
    	mask = tf.greater(cur_x, 0)
		cur_x = tf.boolean_mask(cur_x, mask)
		upper_data.append(cur_x)

	upper_data = tf.concat(upper_data, 0)

  	self.data = upper_data
  	self.num_asset = data.shape[1]
  	self.rolling_wind = rolling_window
  	self.x_dim = num_asset*(num_asset+1)/2
  	self.hidden_dim = self.x_dim * 20
    self.Wxh = tf.Variable(glorot_init([self.x_dim , self.hidden_dim]))
    self.Whh = tf.Variable(glorot_init([self.hidden_dim, self.hidden_dim]))
    self.Why = tf.Variable(glorot_init([self.hidden_dim, self.x_dim ]))
    self.bh = tf.Variable(glorot_init([self.hidden_dim]))
    self.by = tf.Variable(glorot_init([self.x_dim]))

  def predict(self, X):
  	self.train()
  	self.model(self.data[-self.rolling_wind, :])
  	return

  # Build RNN Graph
  def model(self, X): 
  	X = tf.log(X)
    rolling_wind = X.shape[0]
    pre_h = tf.zeros() 
    y = []
    for i in range(rolling_wind):
    	cur_x = X[i,:]
		h = tf.matmul(cur_x, self.Wxh) + tf.matmul(pre_h, self.Whh) + self.bh
		pre_h = h
		y.append(tf.matmul(h, self.Why) + self.by)
    return y

  def train(self):
	self.inputs = tf.placeholder(tf.float32, [None, self.x_dim], name='inputs')
	self.predict = self.model(self.inputs)
	self.loss = tf.losses.MSE(self.inputs, self.predict)
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
	
	init_vars = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init_vars)
		for epoch in range(1000):
			for i in range(self.data.shape[0]):
				batch = self.data[i:i+self.rolling_wind,:]
				optimizer.run(feed_dict={inputs:batch})
				if i%5 ==0:
					train_loss = self.loss.eval(feed_dict={inputs:batch})
					print('at iter %d, train loss is %f'%(i, train_loss))

		# test_loss = self.loss.eval(feed_dict={inputs:self.data})
		# print('test loss is', test_loss)





