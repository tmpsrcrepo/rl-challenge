from osim.env import ProstheticsEnv
from models.RandomAgent import RandomAgent

env = ProstheticsEnv(visualize=True)

agent = RandomAgent(env)
agent.run()
print(agent.reward)