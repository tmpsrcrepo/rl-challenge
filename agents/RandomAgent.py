from models.Agent import Agent

class RandomAgent(Agent):
	def action(self, observation):
		sample = self.env.action_space.sample()
		print(sample.shape)
		return sample

	def run(self):
		done = False
		observation = self.env.reset()
		localReward = 0
		for i in range(self.iterations):
			action = self.action(observation)
			observation, reward, done, info = self.env.step(action)
			localReward += reward
			if done:
				observation = self.env.reset()
				self.rewards.append(localReward)
				localReward = 0
