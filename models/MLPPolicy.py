import numpy as np
import tensorflow as tf

class MLPPolicy(tf.keras.Model):
	def __init__(self, actionDim, observationDim, hiddenSize, nLayers, gamma, episilon):
		super().__init__()
		self.layers = {}
		self.observationDim = observationDim
		self.actionDim = actionDim
		self.observation = tf.placeholder(tf.float32, shape = (1, observationDim))
		self.gamma = gamma # discount factor
		self.episilon = episilon # 1- episilon and 1+epislon is the range of ratio of cur and prev policy
		# initialize policy model
		self.cur_policy, self.value_func = self.getPolicyModel(hiddenSize, nLayers)
		# initially prev = cur
		self.prev_policy = self.cur_policy
		self.adv = tf.get_variable("advantage", [], initializer=tf.constant_initializer(0.))

	# TODO: customize the initializer
	def getPolicyModel(self, hiddenSize, nLayers):
		mean, std = None, None
		obs = self.observation
		with tf.variable_scope('policy-model'):
			input_ = obs
			for i in range(nLayers):
				name = 'policy-layer%d'%i
				input_ = tf.layers.Dense(
					input_,
					hiddenSize,
					name = name,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					bias_initializer=tf.zeros_initializer())
				self.layers[name] = input_

			policy = tf.layers.Dense(
					input_,
					hiddenSize,
					name = 'policy-mean',
					kernel_initializer=tf.random_normal_initializer())

		with tf.variable_scope("value-model"):
			name = 'value-layer'
			value = tf.layers.Dense(obs, hiddenSize, 
				name = name,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				bias_initializer=tf.zeros_initializer())
		return policy, value

	def predict(self, observation):
		output = self.mean(observation)

		self.adv = self.adv + self.gamma * self.value_func(observation)

	def clipRatio(self, observation):
		ratio = self.cur_policy(observation) / self.prev_policy(observation)
		if ratio < 1 - self.episilon:
			1 - self.episilon
		elif ratio > 1 + self.episilon:
			1 + self.episilon
		else:
			ratio

	# TODO: easy to plugin different loss function
	def updateLoss(self, observation, reward, prediction):
		# update loss function
		# update adv
		adv = self.adv
		ratio = clipRatio(observation)
		# update the cur policy in the end (prev = cur, cur = new)

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


	# def _act(self, observation):
		
