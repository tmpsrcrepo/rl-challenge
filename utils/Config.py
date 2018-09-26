# distribution types
NORMAL = 'NORMAL'
LOGNORMAL = 'LOGNORMAL'

class AgentConfig:
	iterations = 300

class PPOAgent(AgentConfig):
	T = 50
	gamma = 0.99 # discounting factor
	
