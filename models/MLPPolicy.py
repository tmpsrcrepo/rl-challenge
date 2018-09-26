import numpy as np
import tensorflow as tf

class MLPPolicy(tf.keras.Model):
	def __init__(self, num_output):
		super().__init__()
		self.dense1 = tf.keras.layers.Dense(units=256, 
			kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
			activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=num_output,
        	kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
			activation=tf.nn.relu)

	def _mean(self, observation):
		x = self.dense1(observation)
		x = self.dense2(x)
		return x

	def predict(self, observation):
		output = self.mean(observation)
		# return random value w/ mean predicted by MLP

	# def __init__(self, actionDim, observationDim, numLayer):
	# 	self.rewards = [] # rewards
	# 	self.actions = [] # actions
	# 	self.advs = [] # advantages
	# 	self.actionDim = actionDim # output layer of policy model dimension
	# 	self.observationDim = observationDim # input feature dimension
	# 	self.valueModel = tf.
	# 	# two layer MLP
	# 	self.policyFirstLayer, self.actionMean = getPolicyModel(numLayer)
		
	# 	self.actionMean = tf.get_variable("action_mean", shape = [1, 19],
	# 		initializer = xavier_initializer)
	# 	self.actionStd = tf.get_variable("action_std", shape = [1, 19],
	# 		initializer = xavier_initializer)

	# # TODO: configure how many layers and how many nodes per layer
	# def getPolicyModel(numLayers = 2, numHiddenNodes = 64):
	# 	#TODO: share the initializer for each 
	# 	xavier_initializer=tf.contrib.layers.xavier_initializer(uniform=True)
	# 	action = tf.placeholder(tf.float, [None, self.actionDim]) # batch size
	# 	firstLayer = tf.layers.dense(action, numHiddenNodes, 
	# 		name = 'action_first',
	# 		kernel_initializer = xavier_initializer)
	# 	layer = None
	# 	# first layer (action )
	# 	for i in range(1, numLayers):
	# 		layer = tf.layers.dense()

	# 	actionLayer = tf.layers.dense(layer, self.actionDim, name = 'action_mean',
	# 		kernel_initializer = xavier_initializer)
	# 	return firstLayer, actionLayer

	# def _act(self, observation):
		
