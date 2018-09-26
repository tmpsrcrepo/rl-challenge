class Policy:
	# TODO: dimension check (make sure if its a tuple)
	def __init__(self, actionDim, observationDim):
		self.rewards = [] # rewards
		self.actions = [] # actions
		self.advs = [] # advantages
		self.actionDim = actionDim # output layer of policy model dimension
		self.observationDim = observationDim # input feature dimension
		self.valueModel =  None
		self.policyModel = None

	def _act(self, observation):
		pass

	def updateIterations(self, itr):
		self.iterations = itr

	def run(self):
		pass

	def evaluate():
		return self.rewards
