import gym
from gym import spaces

class CHWWrapper(gym.core.Wrapper):
    """
    Warpper for gym-minigrid that converts images to CHW for torch compatibilities

    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self):
        obs = self.env.reset()
        obs['image'] = obs['image'].transpose((2, 0, 1))
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs['image'] = obs['image'].transpose((2, 0, 1))
        return obs, reward, done, info