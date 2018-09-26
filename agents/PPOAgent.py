from models.Agent import Agent
import tensorflow as tf
from utils.Config import *

# continuous action space
class PPOAgent(Agent):
	def __init__(self, env, episilon, episodes, iterations=400):
		self.env = env
		# reward per episode
		self.rewards = []
		self.states = 
		self.iterations = iterations
		self.actions = []
		# continuous action: predict gaussian mean and gaussian std
		self.episodes = episodes
		self.episilon = episilon

	def action(self, observation):
		self.gaussian

	def value(self):


	def getAdvantage(slef):


	def clip(self, loss):
		max()

	def run(self):
		for e in range(0, episodes):
			for i in range(0, iterations):

	# update
	def rollout(self):


