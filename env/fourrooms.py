from gym_minigrid.wrappers import *
from gym_minigrid.minigrid import *
from .wrappers import *
from gym_minigrid.envs.fourrooms import FourRoomsEnv

import random
import math

TI_TEST_TASKS = {'ti-1': ['1-2', '2-1',
                          '1-3', '3-1',
                          '2-4', '4-2',
                          '3-4', '4-3'], # adjacent room navigations
                 'ti-2': ['1-4', '4-1', 
                          '2-3', '3-2'] # diagonal room navigations
                }

class NewFourRoomsEnv(FourRoomsEnv):
    """
    Extends the FourRoomsEnv in gym_minigrid so things go as i want
    """

    def __init__(self, 
                 agent_pos=None, 
                 goal_pos=None, 
                 seed=None):
        
        self.rooms = [1,2,3,4] # upper left, upper right, lower left, lower right
        self.current_seed = seed if seed is not None else random.randint(1,1000000)
        self.current_task = None
        
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        
        super().__init__()# super init calls reset
        # overwriting max_steps to prevent premature flagging of done (esp. with the action wrapper)
        self.max_steps = max_steps=math.inf
    
    def reset(self, new_seed=None):
        # by default remains the same grid
        
        if new_seed is not None:
            self.current_seed = new_seed
            self.seed(self.current_seed)
        obs = super().reset()
        return obs
    
    def change_agent_default_pos(self, new_pos):
        self._agent_default_pos = new_pos
    
    def change_goal_default_pos(self, new_pos):
        self._goal_default_pos = new_pos
        
    def sample_pos(self, room=None):
        # calls place_obj with obj=None only for sampling purposes
        # if need to avoid agent/goal overlap needs to make sure self.agent_pos is set before calling place_obj()
        
        # overwrite seed -- this seed thing is so annoying
        # want to use consistent seed for grid creation but randomize position sampling
        self.seed(random.randint(0, 100000))
        
        pos = None
        
        if room is None: # sample any empty location in grid
            pos = self.place_obj(obj=None, top=(0,0), size=(19,19))
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
        
        self.seed(self.current_seed) # reseed back
            
        return pos
    
    def gen_obs(self, goal=False):
        
        if not goal:
            return super().gen_obs()
        
        else:
            state_grid = self.grid.copy()
            state_grid.set(*self._goal_default_pos, None)
            state_grid.set(*self.agent_pos, Goal())

            # Render the state grid
            state_grid = state_grid.encode()[:,:,:1]
            state_grid = self.re_encode_grid(state_grid)
            
            # Render the goal grid with the true goal
            goal_grid = self.grid.encode()[:,:,:1]
            goal_grid = self.re_encode_grid(goal_grid)

            obs = {
                'image': state_grid,
                'goal': goal_grid,
                'mission': self.mission
            }

            return obs
        
    def re_encode_grid(self, grid):
        # original: 1-empty, 2-wall, 8-goal
        # new: 0-empty, 1-wall, 2-goal
        grid -= 1
        grid[grid == 7] = 2
        return grid
    
    def render(self, goal=False, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS):
        
        if not goal:
            
            return super().render(mode=mode, close=close, highlight=highlight, tile_size=tile_size)
        
        else:
            '''
            hacks super().render() and generate separate current state and goal state images
            '''

            if close:
                if self.window:
                    self.window.close()
                return

            if mode == 'human' and not self.window:
                import gym_minigrid.window
                self.window = gym_minigrid.window.Window('gym_minigrid')
                self.window.show(block=False)

            # Compute which cells are visible to the agent
            _, vis_mask = self.gen_obs_grid()

            # Compute the world coordinates of the bottom-left corner
            # of the agent's view area
            f_vec = self.dir_vec
            r_vec = self.right_vec
            top_left = self.agent_pos + f_vec * (self.agent_view_size-1) - r_vec * (self.agent_view_size // 2)

            # Mask of which cells to highlight
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

            # For each cell in the visibility mask
            for vis_j in range(0, self.agent_view_size):
                for vis_i in range(0, self.agent_view_size):
                    # If this cell is not visible, don't highlight it
                    if not vis_mask[vis_i, vis_j]:
                        continue

                    # Compute the world coordinates of this cell
                    abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                    if abs_i < 0 or abs_i >= self.width:
                        continue
                    if abs_j < 0 or abs_j >= self.height:
                        continue

                    # Mark this cell to be highlighted
                    highlight_mask[abs_i, abs_j] = True
            
            '''
            hacky way of aligning state and goal observation:
            remder a pseudo 'goal' at agent_pos, and another real 'goal' at goal_pos
            all to sideway gym-minigrid's smart handling of rendering agent or goal objects
            '''
            state_grid = self.grid.copy()
            state_grid.set(*self._goal_default_pos, None)
            state_grid.set(*self.agent_pos, Goal())

            # Render the state grid
            state_img = state_grid.render( # with agent_pos replaced with a goal block (so that it looks green :)
                tile_size,
                agent_pos=None,
                agent_dir=None,
                highlight_mask=highlight_mask if highlight else None
            )
            
            # Render the goal grid
            goal_img = self.grid.render( # with the true goal
                tile_size,
                agent_pos=None,
                agent_dir=None,
                highlight_mask=highlight_mask if highlight else None
            )

            if mode == 'human':
                self.window.show_img(img)
                self.window.set_caption(self.mission)
            
            return state_img, goal_img

class FourRoomsTask:
    
    """
    wraps the the four room environment 
    and can cast the navigation task as a
    transitive inference task (TIT, train/test split)
    """
    
    def __init__(self, 
                 task_type='ti-1', 
                 reward_type='sparse', 
                 goalcond=False, 
                 seed=None):
        
        # task = ti: transitive inference,
        #        random: the default minigrid position sampling
        # seed, this task overwrites position sampling so seed is really just for grid
        
        self.task_type = task_type
        self.reward_type = reward_type
        self.goalcond = goalcond
        self.grid_seed = random.randint(1, 100000) if seed is None else seed
        env = NewFourRoomsEnv()
        
        # apply some default env wrappers
        env = ActionWrapper(env) # 0-up, 1-right, 2-down, 3-left
        if self.goalcond:
            env = GoalCondObsWrapper(env) # add goal field
            env = GoalCondCHWWrapper(env)
        else:
            env = CHWWrapper(env)
        
        self.env = env
        self.agent_ini_pos = None
        self.goal_pos = None
        self.phase = 'train'
        
        self.reset()
    
    def reset(self, new_task=True, new_seed=None, **task_kwargs):
        
        if new_task:
            # sample new start/goal pos
            self.update_task(**task_kwargs)
        self.grid_seed = self.grid_seed if new_seed is None else new_seed
        self.env.seed(self.grid_seed)
        obs = self.env.reset()
        return obs
    
    def step(self, a):
        ns, r, done, info = self.env.step(a)
        # overwrite reward
        r = self.reward(done)
        return ns, r, done, info
    
    def reward(self, done):
        if self.reward_type=='sparse':
            r = 0.0 if not done else 1.0
        elif self.reward_type=='dense':
            r = -1.0 if not done else 0.0
        return r
    
    def set_phase(self, phase):
        self.phase = phase
    
    def update_task(self, max_tries=1000):
        
        if self.task_type == 'random':
            self.agent_ini_pos = self.env.sample_pos()
            self.goal_pos = self.env.sample_pos()
            
        elif self.task_type == 'fixed':
            # fixed start and goal
            if self.agent_ini_pos is None: # and self.goal_pos is None
                r1, r2 = random.sample(self.env.rooms, 2)
                self.agent_ini_pos = self.env.sample_pos(r1)
                self.goal_pos = self.env.sample_pos(r2)
        
        elif self.task_type == 'fixed-g':
            # fixed goal, changing start
            if self.goal_pos is None: # and self.goal_pos is None
                self.goal_pos = self.env.sample_pos()
            self.agent_ini_pos = self.env.sample_pos()
        
        elif self.task_type == 'fixed-s':
            # fixed start, changing goal
            if self.agent_ini_pos is None:
                self.agent_ini_pos = self.env.sample_pos()
            self.goal_pos = self.env.sample_pos()
        
        elif 'ti' in self.task_type:
            # phase = train/test
            # 'ti-1': train with within-room, test with across-room
            # 'ti-2': train with within or adjacent room, test with diagonal room

            self.test_tasks = TI_TEST_TASKS[self.task_type]

            if self.phase=='train':
                # construct a task that doesn't overlap with test tasks
                num_tries = 0
                while True:
                    if num_tries > max_tries:
                        # raise RecursionError('rejection sampling failed in FourRoomsTask.update_task()')
                        print('rejection sampling failed in FourRoomsTask.update_task()')
                        break
                    num_tries += 1

                    if self.task_type=='ti-1':
                        r1 = random.choice(self.env.rooms)
                        r2 = r1
                    if self.task_type=='ti-2':
                        r1 = random.choice(self.env.rooms)
                        r2 = random.choice(self.env.rooms)
                    task = '%d-%d' % (r1, r2)
                    if task not in self.test_tasks:
                        break

                self.current_task = task

            if self.phase=='test':
                self.current_task = random.choice(self.test_tasks)

            # modify agent/goal positions based on current task
            start_room = int(self.current_task[0])
            goal_room = int(self.current_task[-1])

            self.agent_ini_pos = self.env.sample_pos(start_room)
            self.goal_pos = self.env.sample_pos(goal_room)
            
        self.env.change_agent_default_pos(self.agent_ini_pos)
        self.env.change_goal_default_pos(self.goal_pos)