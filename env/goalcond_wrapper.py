import gym
from gym import spaces

class GoalCondWrapper(gym.core.Wrapper):
    """
    Warpper for gym-minigrid that generates separate observation for agent and goal

    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self):
        pass
    
    def step(self, action):
        pass