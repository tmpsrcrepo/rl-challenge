import numpy as np
import tensorflow as tf

class MLPPolicy(tf.keras.Model):
	def __init__(self, actionDim, observationDim, hiddenSize, nLayers, gamma, episilon, T):
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
		self.min_bound = 1 - self.episilon
		self.max_bound = 1 + self.episilon
		self.rewards = []
		self.values = []
		self.prev_policies = []
		self.cur_policies = []
		self.observations = []
		self.T = T
		self.sigma = tf.get_variable("std", (1, self.actionDim), initializer=tf.constant_initializer(0.))
		# TODO: adaptive learning rate / plugin optimizer
		self.optimizezr = tf.train.AdamOptimizer(learning_rate = 0.001)

	# TODO: customize the initializer
	def getPolicyModel(self, hiddenSize, nLayers):
		policy, value = None, None
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
					self.actionDim,
					name = 'policy-mean',
					kernel_initializer=tf.random_normal_initializer())

		# TODO: dist model
		with tf.variable_scope("value-model"):
			name = 'value-layer'
			value = tf.layers.Dense(self.observation, 1, 
				name = name,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				bias_initializer=tf.zeros_initializer())
		return policy, value

	# TODO: customize -> use different distributions
	def normDist(mean):
		return np.random.normal(mean, self.sigma)

	def run(self):
		with tf.Session() as sess:
			observation = self.env.reset()
			for t in range(self.T):
				self.observations.append(observation)
				cur_mean, pre_mean,value = sess.run(
					[self.cur_policy(self.observation), self.prev_policy(self.observation), self.value_func(self.observation)],
					feed_dict = {self.observation: observation})
				cur_action = self.normDist(cur_mean)
				prev_action = self.normDist(pre_mean)
				observation, reward, done, info = self.env.step(action)
				self.cur_policies.append(cur_action)
				self.prev_policies.append(prev_action)
				self.values.append(value)
				self.rewards.append(reward)

			loss = self.calculateLoss()
			self.prev_policy = self.cur_policy
			sess.run(self.optimizezr.minimize(loss))

	# TODO: easy to plugin different loss function
	def calculateLoss(self):
		# update loss function
		# update adv
		R = 0
		loss = 0
		for t in range(self.T - 1, -1, -1):
			reward = self.rewards[t]
			observation = self.observations[t]
			cur = self.cur_policy(observation)
			pre = self.prev_policy(observation)
			ratio = tf.clip_by_value(cur/pre, self.min_bound, self.max_bound)
			R += reward + self.gamma * self.value_func(observation)
			adv = R - self.value_func(observation)
			loss += min(cur/pre, ratio) * adv

		self.rewards = []
		self.values = []
		self.cur_policies = []
		self.prev_policies = []
		self.observations  = []
		return loss
		# update the cur policy in the end (prev = cur, cur = new)
