from osim.env import ProstheticsEnv
from agents.RandomAgent import RandomAgent

env = ProstheticsEnv(visualize=True)

agent = RandomAgent(env)
# agent.run()
# print(agent.reward)

# init = tf.global_variables_initializer()
# with tf.Session() as sess:
# 	sess.run(init)