from env import FourRoomsTask, EmptyGridTask
from memory import ReplayBuffer, RemergeMemory
from agent import DQN

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

def train(task,
          agent,
          writer,
          use_remerge=False,
          replan_prob=0.5,
          switch_goal_prob=0.5,
          n_level=1,
          num_epochs=2e4,
          max_steps=50,
          test_max_steps=50,
          test_interval=250,
          target_update_interval=20,
          checkpoint_interval=5e3):
    
    # fill memory with random experiences
    test(task, agent, num_episodes=100, use_remerge=False, max_steps=max_steps, remember=True, explore=True)
    
    for i in range(1, int(num_epochs)+1):
    
        obs = task.reset()
        state = torch.from_numpy(obs['state'].astype(np.float32)).unsqueeze(0).to(agent.device)
        goal = torch.from_numpy(obs['goal'].astype(np.float32)).unsqueeze(0).to(agent.device)

        losses = []

        for t in range(max_steps):  # total of num_epochs x max_steps updates
        
            action = agent.step(state=state, goal=goal, explore=True)[0]
            next_obs, reward, done, _ = task.step(action.item())
            next_state = torch.from_numpy(next_obs['state'].astype(np.float32)).unsqueeze(0).to(agent.device)
            # goal shouldn't change within an episode
            reward = torch.tensor([reward], dtype=torch.float, device=agent.device)
            
            if done:
                next_state = None

            agent.memory.add(state, action, reward, next_state, goal)

            loss = agent._train()
            if loss is not None:
                losses.append(loss.item())
                
            state = next_state
            
            if done:
                break
        
        agent.step_count += 1 # update exploration threshold by epoch
        
        if writer is not None:
            writer.add_scalar('loss', np.nanmean(losses), i)
        else:
            print('epoch: ', i, '   loss=%.6f'%np.nanmean(losses))

        if i % target_update_interval == 0:
            agent.target_network.load_state_dict(agent.network.state_dict())
        
        if i % test_interval == 0:
            
            # -- training task --
            test_results = test(task, agent,
                                use_remerge=False,
                                num_episodes=20, max_steps=test_max_steps,
                                remember=False, explore=False)
            if writer is not None:
                writer.add_scalar('avg_n_step_train', np.mean(test_results['n_step']), i)
                writer.add_scalar('avg_solved_train', np.mean(test_results['solved']), i)
                writer.add_scalar('avg_reward_train', np.mean(test_results['reward']), i)
                writer.add_scalar('avg_excess_steps_train', np.mean(test_results['excess_steps']), i)
            else:
                print('\t test avg_n_step_train: ', np.mean(test_results['n_step']))
                print('\t test avg_solved_train: ', np.mean(test_results['solved']))
                print('\t test avg_reward_train: ', np.mean(test_results['reward']))
                print('\t test avg_excess_step_train: ', np.mean(test_results['excess_steps']))
            
            # -- test task (may require generalization) --
            task.set_phase('test')
            test_results = test(task, agent,
                                use_remerge=use_remerge, 
                                replan_prob=replan_prob,
                                switch_goal_prob=switch_goal_prob,
                                n_level=n_level,
                                num_episodes=20, max_steps=test_max_steps,
                                remember=False, explore=False)
            if writer is not None:
                writer.add_scalar('avg_n_step_test', np.mean(test_results['n_step']), i)
                writer.add_scalar('avg_solved_test', np.mean(test_results['solved']), i)
                writer.add_scalar('avg_reward_test', np.mean(test_results['reward']), i)
                writer.add_scalar('avg_excess_steps_test', np.mean(test_results['excess_steps']), i)
            else:
                print('\t test avg_n_step_test: ', np.mean(test_results['n_step']))
                print('\t test avg_solved_test: ', np.mean(test_results['solved']))
                print('\t test avg_reward_test: ', np.mean(test_results['reward']))
                print('\t test avg_excess_step_train: ', np.mean(test_results['excess_steps']))
            
            task.set_phase('train')

        if i % checkpoint_interval == 0: # gotta save the last timepoint...
            torch.save(agent.network.state_dict(), writer.get_logdir()+'/checkpoint_%d'%i)

def test(task,
         agent,
         use_remerge=False,
         replan_prob=0.5,
         switch_goal_prob=0.5,
         n_level=1,
         num_episodes=20,
         max_steps=50,
         remember=False,
         explore=True):
    
    # can be used to fill buffer with experiences in the training distribution
    # if task.phase is flagged 'train'

    results = {'n_step':np.zeros(num_episodes),
               'solved':np.zeros(num_episodes),
               'reward':np.zeros(num_episodes),
               'excess_steps':np.zeros(num_episodes)}

    # make copies of the same task (task phase carried over)
    task_list = [task.clone() for i in range(num_episodes)]
    remaining_tasks = list(range(num_episodes)) # all tasks remains to be done
    
    # reset obs
    obs_list = [task.reset() for task in task_list]
    states = np.array([obs['state'] for obs in obs_list])
    states = torch.from_numpy(states.astype(np.float32)).to(agent.device)
    goals = np.array([obs['goal'] for obs in obs_list])
    goals = torch.from_numpy(goals.astype(np.float32)).to(agent.device)
    
    plans = [None] * len(remaining_tasks)
    switch_probs = np.linspace(switch_goal_prob, 0, num=max_steps//2).tolist() + [0.0] * (max_steps-max_steps//2)
    
    if use_remerge:
        agent.memory.wire_memory()
    
    for t in range(max_steps):
        
        if len(remaining_tasks)==0:
            break
        
        modified_goals = goals.clone()

        if use_remerge:
            for i in range(len(remaining_tasks)):
                # with some probability replace the goal with a subgoal
                if random.random() < switch_probs[t]:
                    # with some probability generate new subgoal
                    if random.random() < replan_prob:
                        plans[i] = agent.memory.plan(s_probe=states[i,...].detach().cpu().numpy(),
                                                     ns_probe=goals[i,...].detach().cpu().numpy(),
                                                     n_level=n_level)
                    if plans[i] is not None:
                        subgoal = plans[i][random.choice([0,1])]
                        modified_goals[i] = torch.from_numpy(np.array(subgoal).astype(np.float32))
        
        with torch.no_grad():
            actions = agent.step(state=states, goal=modified_goals, explore=explore)
        next_obs_list = [task_list[task_num].step(actions[i].item()) for i, task_num in enumerate(remaining_tasks)] # (obs,r,done,_)
        next_states = np.array([obs[0]['state'] for obs in next_obs_list])
        next_states = torch.from_numpy(next_states.astype(np.float32)).to(agent.device)
        
        # TODO: goals shouldn't need to change
        goals = np.array([obs[0]['goal'] for obs in next_obs_list])
        goals = torch.from_numpy(goals.astype(np.float32)).to(agent.device)
        
        if remember:
            for i in range(len(remaining_tasks)): # collected this many transitions
                state = states[i:i+1]
                action = actions[i]
                reward = torch.tensor([next_obs_list[i][1]], dtype=torch.float, device=agent.device)
                next_state = next_states[i:i+1]
                # but if done overwrite next_state
                if next_obs_list[i][2]:
                    next_state = None
                goal = goals[i:i+1]
                agent.memory.add(state, action, reward, next_state, goal)

        # check performance
        current_remaining_tasks = remaining_tasks.copy()
        for i, task_num in enumerate(current_remaining_tasks):
            results['n_step'][task_num] = t+1
            results['solved'][task_num] = next_obs_list[i][2] # done
            results['reward'][task_num] = next_obs_list[i][1] # reward at this step
            results['excess_steps'][task_num] = (t+1) - next_obs_list[i][0]['optim_step']
            if results['solved'][task_num]: # if done for this task
                remaining_tasks.remove(task_num)
                next_states = torch.cat((next_states[:i], next_states[i+1:]))
                goals = torch.cat((goals[:i], goals[i+1:]))
                plans = plans[:i] + plans[i+1:]
        
        # Move to the next state
        states = next_states

    return results

def run(log_dir,
        cuda_idx,
        task_type,
        reward_type,
        memory,
        remerge_size,
        train_max_steps,
        test_max_steps,
        replan_prob,
        switch_goal_prob,
        n_level):
    
    device = torch.device("cuda:%d" % cuda_idx)
    
    run_key = '{}_{}_{}'.format(task_type, reward_type, memory)
    if memory == 'remerge':
        run_key += '_size{}_preplan{}_pswitch{}'.format(remerge_size, replan_prob, switch_goal_prob)
    run_key += '_{}'.format(datetime.now().strftime('%y%m%d%H%M'))

    writer = SummaryWriter(log_dir+run_key)
    task = EmptyGridTask(size=19,
                         task_type=task_type,
                         reward_type=reward_type,
                         goalcond=True)
    
    use_remerge = True if memory == 'remerge' else False
    memory = RemergeMemory(hidden_size=remerge_size, state_size=remerge_size) if memory == 'remerge' else ReplayBuffer()
    agent = DQN(state_dim=4, # state_dim=obs['image'].shape, 
                action_dim=4, 
                goalcond=True, 
                device=device, 
                memory=memory)
    
    train(task,
          agent, 
          writer=writer, 
          max_steps=train_max_steps, 
          test_max_steps=test_max_steps,
          use_remerge=use_remerge,
          replan_prob=replan_prob,
          switch_goal_prob=switch_goal_prob,
          n_level=n_level)
    
    print('two thousand years later...')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_dir', help='logging directory', default='/data5/liyuxuan/remerge/')
    parser.add_argument('--cuda_idx', help='gpu to use', type=int, default=4)
    parser.add_argument('--task', help='task to run', default='custom')
    parser.add_argument('--reward', help='reward to use', default='euclidean')
    parser.add_argument('--memory', help='memory to use', default='regular')
    parser.add_argument('--remerge_size', type=int, default=1000)
    parser.add_argument('--train_max_steps', type=int, default=50)
    parser.add_argument('--test_max_steps', type=int, default=50)
    parser.add_argument('--replan_prob', type=float, default=0.5)
    parser.add_argument('--switch_goal_prob', type=float, default=0.5)
    parser.add_argument('--n_level', type=int, default=1)
    args = parser.parse_args()
    run(log_dir=args.log_dir,
        cuda_idx=args.cuda_idx,
        task_type=args.task,
        reward_type=args.reward,
        memory=args.memory,
        remerge_size=args.remerge_size,
        train_max_steps=args.train_max_steps,
        test_max_steps=args.test_max_steps,
        replan_prob=args.replan_prob,
        switch_goal_prob=args.switch_goal_prob,
        n_level=args.n_level)