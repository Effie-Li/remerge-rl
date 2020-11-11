import gym
import numpy as np
from gym import spaces
from gym_minigrid.wrappers import *

class ActionWrapper(gym.core.Wrapper):
    """
    Warpper for gym-minigrid that uses up/down/left/right actions

    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(4) # rewrite action space

    def reset(self):
        self.env.reset()
        self.env.agent_dir = 0 # always face right
        return self.env.gen_obs()
    
    def step(self, action):

        if action==0: # move up
            self.env.agent_dir = 3 # face up
            obs, reward, done, info = self.env.step(2) # move forward (move up)
            self.env.agent_dir = 0 # back to face right
        elif action==1: # move right
            obs, reward, done, info = self.env.step(2) # move forward
        elif action==2: # move down
            self.env.agent_dir = 1 # face down
            obs, reward, done, info = self.env.step(2) # move forward (move down)
            self.env.agent_dir = 0 # back to face right
        elif action==3: # move left
            self.env.agent_dir = 2 # face left
            obs, reward, done, info = self.env.step(2) # move forward (move up)
            self.env.agent_dir = 0 # back to face right

        return obs, reward, done, info

class CHWWrapper(gym.core.Wrapper):
    """
    Warpper for gym-minigrid that converts images to CHW for torch compatibilities
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs = self.env.reset()
        obs['image'] = obs['image'].transpose((2, 0, 1))
        return obs
    
    def step(self, action):
        
        obs, reward, done, info = self.env.step(action)
        obs['image'] = obs['image'].transpose((2, 0, 1))
        
        return obs, reward, done, info
    
class GoalCondCHWWrapper(gym.core.Wrapper):
    """
    Warpper for gym-minigrid that converts (both state and goal) images to CHW for torch compatibilities
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs = self.env.reset()
        obs['image'] = obs['image'].transpose((2, 0, 1))
        obs['goal'] = obs['goal'].transpose((2, 0, 1))
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs['image'] = obs['image'].transpose((2, 0, 1))
        obs['goal'] = obs['goal'].transpose((2, 0, 1))
        return obs, reward, done, info
    
class GoalCondRGBImgObsWrapper(RGBImgObsWrapper):
    """
    Warpper for gym-minigrid that add a 'goal' field in the observation

    """
    
    def __init__(self, env, img_dim='chw'):
        super().__init__(env)
        self.observation_space = self.env.observation_space
        self.observation_space.spaces['goal'] = self.observation_space.spaces['image']
    
    def observation(self, obs):
        env = self.unwrapped

        rgb_img, goal_img = env.render( # must be able to unpack two images
            goal=True,
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )

        return {
            'mission': obs['mission'],
            'image': rgb_img,
            'goal': goal_img
        }
    
class GoalCondObsWrapper(gym.core.ObservationWrapper):
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space
        self.observation_space.spaces['goal'] = self.observation_space.spaces['image']
        
    def observation(self, obs):
        
        obs = self.env.gen_obs(goal=True)
        
        return obs