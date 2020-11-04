from gym_minigrid.wrappers import *
from gym_minigrid.minigrid import *
from gym_minigrid.envs.fourrooms import FourRoomsEnv
from . import ActionWrapper

import random
import math

TI_TEST_TASKS = {'diff-1': ['1-2', '2-1',
                            '1-3', '3-1',
                            '2-4', '4-2',
                            '3-4', '4-3'], # adjacent room navigations
                 'diff-2': ['1-4', '4-1', 
                            '2-3', '3-2'] # diagonal room navigations
                }

class NewFourRoomsEnv(FourRoomsEnv):
    """
    Extends the FourRoomsEnv in gym_minigrid so things go as i want
    """

    def __init__(self, agent_pos=None, goal_pos=None, seed=None):
        
        self.rooms = [1,2,3,4] # upper left, upper right, lower left, lower right
        self.default_seed = seed if seed is not None else random.randint(1,1000000)
        self.current_task = None
        
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__() # calls reset
    
    def reset(self, new_seed=None):
        # by default remains the same grid
        seed = new_seed if new_seed is not None else self.default_seed
        self.seed(seed)
        obs = super().reset()
        return obs
    
    def change_agent_default_pos(self, new_pos):
        self._agent_default_pos = new_pos
    
    def change_goal_default_pos(self, new_pos):
        self._goal_default_pos = new_pos
        
    def sample_pos(self, room=None):
        # calls place_obj with obj=None only for sampling purposes
        # if need to avoid agent/goal overlap needs to make sure self.agent_pos is set before calling place_obj()
        
        pos = None
        
        if room is None: # sample any empty location in grid
            pos = self.place_obj(obj=None)
        elif room == 1:
            pos = self.place_obj(obj=None, top=(0,0), size=(9,9))
        elif room == 2:
            pos = self.place_obj(obj=None, top=(0,9), size=(9,9))
        elif room == 3:
            pos = self.place_obj(obj=None, top=(9,0), size=(9,9))
        elif room == 4:
            pos = self.place_obj(obj=None, top=(9,9), size=(9,9))
        else:
            raise ValueError('which room???')
            
        return pos

class FourRoomsTask:
    
    """
    wraps the the four room environment 
    and can cast the navigation task as a
    transitive inference task (TIT, train/test split)
    """
    
    def __init__(self, task_type='ti', seed=None):
        
        # task = ti: transitive inference,
        #        random: the default minigrid position sampling
        # seed, this task overwrites position sampling so seed is really just for grid
        
        self.task_type = task_type
        self.grid_seed = random.randint(1, 100000) if seed is None else seed
        self.env = NewFourRoomsEnv()
        
        # apply some default env wrappers
        self.env = ActionWrapper(self.env) # 0-up, 1-right, 2-down, 3-left
        self.env = RGBImgObsWrapper(self.env) # Get pixel observations
        self.env = ImgObsWrapper(self.env)
        
        self.reset()
    
    def reset(self, new_task=True, new_seed=None, **task_kwargs):
        
        if new_task:
            # sample new start/goal pos
            self.update_task(**task_kwargs)
        self.grid_seed = self.grid_seed if new_seed is None else new_seed
        self.env.seed(self.grid_seed)
        obs = self.env.reset()
        return obs
    
    def step(self, action):
        ns, r, done, info = self.env.step(a)
        return ns, r, done, info
    
    def update_task(self, phase='train', difficulty=1, max_tries=1000):
        
        if self.task_type == 'random':
            self.env.agent_pos = self.env.sample_pos()
            self.env.change_agent_default_pos(self.env.agent_pos)
            self.env.goal_pos = self.env.sample_pos()
            self.env.change_goal_default_pos(self.env.goal_pos)
        
        elif self.task_type == 'ti':
            # phase = train/test
            # difficulty = 1/2 (1: train with within-room, test with across-room)
            #                  (2: train with within or adjacent room, test with diagonal room)

            self.test_tasks = TI_TEST_TASKS['diff-%d'%difficulty]

            if phase=='train':
                # construct a task that doesn't overlap with test tasks
                num_tries = 0
                while True:
                    if num_tries > max_tries:
                        raise RecursionError('rejection sampling failed in FourRoomsTask.update_task()')
                    num_tries += 1

                    if difficulty==1:
                        r1 = random.choice(self.env.rooms)
                        r2 = r1
                    if difficulty==2:
                        r1 = random.choice(self.env.rooms)
                        r2 = random.choice(self.env.rooms)
                    task = '%d-%d' % (r1, r2)
                    if task not in self.test_tasks:
                        break

                self.current_task = task

            if phase=='test':
                self.current_task = random.choice(self.test_tasks)

            # modify agent/goal positions based on current task
            start_room = int(self.current_task[0])
            goal_room = int(self.current_task[-1])
            
            self.env.agent_pos = None
            self.env.goal_pos = None
            self.env.agent_pos = self.env.sample_pos(start_room)
            self.env.change_agent_default_pos(self.env.agent_pos)
            self.env.goal_pos = self.env.sample_pos(goal_room)
            self.env.change_goal_default_pos(self.env.goal_pos)