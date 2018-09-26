class Agent:
	def __init__(self, env, iterations=20):
			self.env = env
			# reward per episode
			self.rewards = []
			self.iterations = iterations

	def action(self, observation):
		pass

	def updateIterations(self, itr):
		self.iterations = itr

	def run(self):
		pass

	def evaluate():
		return self.rewards
