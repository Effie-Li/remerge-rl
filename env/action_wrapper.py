import gym
from gym import spaces

class ActionWrapper(gym.core.Wrapper):
    """
    Warpper for gym-minigrid that uses up/down/left/right actions

    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = spaces.Discrete(4) # rewrite action space

    def reset(self):
        self.env.reset()
        self.env.agent_dir = 0 # always face right
        return self.env.gen_obs()
    
    def step(self, action):

        if action==0: # move up
            self.env.step(0) # turn left
            obs, reward, done, info = self.env.step(2) # move forward (move up)
            if not done:
                obs, reward, done, info = self.env.step(1) # turn back to face right
        elif action==1: # move right
            obs, reward, done, info = self.env.step(2) # move forward
        elif action==2: # move down
            self.env.step(1) # turn right
            obs, reward, done, info = self.env.step(2) # move forward (move down)
            if not done:
                obs, reward, done, info = self.env.step(0) # turn back to face right
        elif action==3: # move left
            self.env.step(0)
            self.env.step(0) # turn twice to face left
            obs, reward, done, info = self.env.step(2) # move forward (move up)
            if not done:
                self.env.step(1)
                obs, reward, done, info = self.env.step(1) # turn back to face right

        return obs, reward, done, info
