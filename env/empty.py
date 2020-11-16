from gym_minigrid.minigrid import *
from .wrappers import *
from gym_minigrid.envs.empty import EmptyEnv

import random
import math

class NewEmptyGridEnv(EmptyEnv):
    """
    Extends the EmptyEnv in gym_minigrid so things go as i want
    """

    def __init__(self, 
                 agent_pos=None, 
                 goal_pos=None,
                 size=8, 
                 seed=None):
        
        self.current_seed = seed if seed is not None else random.randint(1,1000000)
        self.size = size
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        
        super().__init__(size=size,
                         agent_start_pos = self._agent_default_pos)# super init calls reset
        # overwriting max_steps to prevent premature flagging of done (esp. with the action wrapper)
        self.max_steps = math.inf
    
    def reset(self, new_seed=None):
        
        # by default remains the same grid
        
        if new_seed is not None:
            self.current_seed = new_seed
            self.seed(self.current_seed)
        obs = super().reset()
        return obs
    
    def change_agent_start_pos(self, new_pos):
        self.agent_start_pos = new_pos
    
    def sample_pos(self, room=None):
        # calls place_obj with obj=None only for sampling purposes
        # if need to avoid agent/goal overlap needs to make sure self.agent_pos is set before calling place_obj()
        
        # overwrite seed -- this seed thing is so annoying
        # want to use consistent seed for grid creation but randomize position sampling
        self.seed(random.randint(0, 100000))
        
        pos = None
        half = self.size//2
        
        if room is None: # sample any empty location in grid
            pos = self.place_obj(obj=None, top=(0,0), size=(self.size,self.size))
        elif room == 1:
            pos = self.place_obj(obj=None, top=(0,0), size=(half,half))
        elif room == 2:
            pos = self.place_obj(obj=None, top=(0,half), size=(half,half))
        elif room == 3:
            pos = self.place_obj(obj=None, top=(half,0), size=(half,half))
        elif room == 4:
            pos = self.place_obj(obj=None, top=(half,half), size=(half,half))
        else:
            raise ValueError('which room???')
        
        self.seed(self.current_seed) # reseed back
        
        return pos
    
#     def _gen_grid(self, width, height):
#         # Create an empty grid
#         self.grid = Grid(width, height)

#         # Generate the surrounding walls
#         self.grid.wall_rect(0, 0, width, height)

#         # Place the goal square
#         if self._goal_default_pos is not None:
#             self.put_obj(Goal(), *self._goal_default_pos)

#         # Place the agent
#         if self._agent_default_pos is not None:
#             self.agent_pos = self.agent_start_pos
#         else:
#             self.place_agent()

#         self.mission = "get to the green goal square"
    
#     def gen_obs(self, goal=False):
        
#         if not goal:
#             return super().gen_obs()
        
#         else:
#             state_grid = self.grid.copy()
#             if self._goal_default_pos is not None:
#                 state_grid.set(*self._goal_default_pos, None)
#             else:
#                 # remove default goal
#                 state_grid.set(*(self.size-2, self.size-2), None)
#             state_grid.set(*self.agent_pos, Goal())

#             # Render the state grid
#             state_grid = state_grid.encode()[:,:,:1]
#             state_grid = self.re_encode_grid(state_grid)
            
#             # Render the goal grid with the true goal
#             goal_grid = self.grid.encode()[:,:,:1]
#             goal_grid = self.re_encode_grid(goal_grid)

#             obs = {
#                 'image': state_grid,
#                 'goal': goal_grid,
#                 'mission': self.mission
#             }

#             return obs
    
#     def re_encode_grid(self, grid):
#         # original: 1-empty, 2-wall, 8-goal
#         # new: 0-empty, 1-wall, 2-goal
#         grid -= 1
#         grid[grid == 7] = 2
#         return grid

class EmptyGridTask:
    
    '''
    for now, just uses [x-y] coord as observation
    '''
    
    def __init__(self,
                 size=8,
                 task_type='fixed', 
                 reward_type='sparse',
                 agent_ini_pos=None,
                 goal_pos=None,
                 step_range=(1, 100),
                 goalcond=False):
        
        self.size = size
        self.task_type = task_type
        self.reward_type = reward_type
        self.goalcond = goalcond
        
        self.agent_ini_pos = agent_ini_pos
        self.goal_pos = goal_pos
        
        self.step_range = step_range
        
        env = NewEmptyGridEnv(size=self.size,
                              agent_pos=(2,2),
                              goal_pos=(6,6)) # placeholders
        self.env = ActionWrapper(env)
        
        self.phase = 'train'
    
    def change_step_range(self, _min, _max):
        self.step_range = (_min, _max)
    
    def reset(self, new_task=True, **task_kwargs):
        
        if new_task:
            # sample new start/goal pos, adapted for each task_type
            self.update_task(**task_kwargs) # sets self.agent_ini_pos and self.goal_pos
        self.env.change_agent_start_pos(self.agent_ini_pos)
        self.env.reset()
        obs = {'state':np.array(self.env.agent_pos),
               'goal':np.array(self.goal_pos),
               'optim_step':self.optim_step}
        return obs
    
    def step(self, a):
        ns, r, done, info = self.env.step(a)
        # overwrite reward
        done = self.arrived()
        r = self.reward(done)
        ns = {'state':np.array(self.env.agent_pos),
              'goal':np.array(self.goal_pos),
              'optim_step':self.optim_step}
        return ns, r, done, info
    
    def arrived(self):
        agent_pos = self.env.agent_pos
        goal_pos = self.goal_pos
        if (agent_pos[0]==goal_pos[0]) and (agent_pos[1]==goal_pos[1]):
            return True
        else:
            return False
    
    def reward(self, done):
        if self.reward_type=='sparse':
            r = 0.0 if not done else 1.0
        elif self.reward_type=='dense':
            r = -1.0 if not done else 0.0
        elif self.reward_type=='euclidean':
            agent_pos = np.array(self.env.agent_pos)
            goal_pos = np.array(self.goal_pos)
            distance = np.sqrt(np.sum((agent_pos-goal_pos)**2))
            r = -distance
        return r
    
    def set_phase(self, phase):
        self.phase = phase
    
    def update_task(self, max_tries=1000):
        
        # generate a task that at optim takes [min_step, max_step] steps
        
        if self.task_type == 'ti':
            # imagine rooms

            if self.phase=='train':
                r1 = random.choice([1,2,3,4])
                r2 = r1
            else: # self.phase=='test':
                r1, r2 = random.sample([1,2,3,4],2)
            
            self.agent_ini_pos = self.env.sample_pos(r1)
            self.goal_pos = self.env.sample_pos(r2)
            self.optim_step = sum(abs(np.array(self.agent_ini_pos)-np.array(self.goal_pos)))
            
            return
        
        for t in range(max_tries):
            
            if self.task_type == 'custom':
                self.optim_step = sum(abs(np.array(self.agent_ini_pos)-np.array(self.goal_pos)))
                return
            
            if 'random' in self.task_type:
                if len(self.task_type) > len('random'):
                    step_cuts = self.task_type.split('-')[1:]
                    if self.phase=='train':
                        self.change_step_range(int(step_cuts[0]), int(step_cuts[1]))
                    else: # self.phase=='test':
                        self.change_step_range(int(step_cuts[2]), int(step_cuts[3]))
                
                self.agent_ini_pos = self.env.sample_pos()
                self.goal_pos = self.env.sample_pos()

            elif self.task_type == 'fixed':
                # fixed start and goal
                if self.agent_ini_pos is None: # and self.goal_pos is None
                    self.agent_ini_pos = self.env.sample_pos()
                    self.goal_pos = self.env.sample_pos()

            elif self.task_type == 'fixed-g':
                # fixed goal, changing start
                if self.goal_pos is None:
                    self.goal_pos = self.env.sample_pos()
                self.agent_ini_pos = self.env.sample_pos()

            elif self.task_type == 'fixed-s':
                # fixed start, changing goal
                if self.agent_ini_pos is None:
                    self.agent_ini_pos = self.env.sample_pos()
                self.goal_pos = self.env.sample_pos()
            
            # check if task is within difficulty
            self.optim_step = sum(abs(np.array(self.agent_ini_pos)-np.array(self.goal_pos)))
            if self.optim_step >= self.step_range[0] and self.optim_step <= self.step_range[1]:
                break
            
            self.agent_ini_pos = None
            self.goal_pos = None
        
    def clone(self):
        # return a copy of the task
        x = EmptyGridTask(size=self.size,
                          task_type=self.task_type, 
                          reward_type=self.reward_type, 
                          goalcond=self.goalcond)
        x.agent_ini_pos = self.agent_ini_pos
        x.goal_pos = self.goal_pos
        x.phase = self.phase
        return x