from env import FourRoomsTask
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
          num_epochs=1e6, 
          max_steps=100,
          log_interval=10,
          test_interval=100,
          target_update_interval=100,
          checkpoint_interval=1e4):
    
    for i in range(int(num_epochs)):

        obs = task.reset()
        state = torch.from_numpy(obs['image'].astype(np.float32)).unsqueeze(0).to(agent.device)
        goal = torch.from_numpy(obs['goal'].astype(np.float32)).unsqueeze(0).to(agent.device)

        losses = []

        for t in range(max_steps):
            action = agent.step(state=state, goal=goal)
            next_obs, reward, done, _ = task.step(action.item())
            next_state = torch.from_numpy(next_obs['image'].astype(np.float32)).unsqueeze(0).to(agent.device)
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

        if i % target_update_interval == 0:
            agent.target_network.load_state_dict(agent.network.state_dict())

        if i % log_interval == 0:
            if len(losses) > 0:
                writer.add_scalar('avg_loss', np.nanmean(losses), i)
        
        if i % test_interval == 0:
            
            # -- training task test --
            
            # test_results = test(task, num_epochs=10) # too slow
            
            test_results = tuple(map(lambda task: test(task, agent, num_episodes=1), 
                                     [task] * 10))
            
            batch_n_step = [x['n_step'][0] for x in test_results]
            batch_solved = [x['solved'][0] for x in test_results]
            batch_reward = [x['reward'][0] for x in test_results]
            
            writer.add_scalar('avg_n_step', np.mean(batch_n_step), i)
            writer.add_scalar('avg_solved', np.mean(batch_solved), i)
            writer.add_scalar('avg_reward', np.mean(batch_reward), i)
            
            # -- transitive inference test --
            task.set_phase('test')
            
            test_results = tuple(map(lambda task: test(task, agent, num_episodes=1), 
                                     [task] * 10))
            
            batch_n_step = [x['n_step'][0] for x in test_results]
            batch_solved = [x['solved'][0] for x in test_results]
            batch_reward = [x['reward'][0] for x in test_results]
            
            writer.add_scalar('avg_n_step_ti', np.mean(batch_n_step), i)
            writer.add_scalar('avg_solved_ti', np.mean(batch_solved), i)
            writer.add_scalar('avg_reward_ti', np.mean(batch_reward), i)
            
            task.set_phase('train')

        if i % checkpoint_interval == 0:
            torch.save(agent.network.state_dict(), writer.get_logdir()+'/checkpoint_%d'%i)

def test(task,
         agent,
         num_episodes=100, 
         max_steps=100,
         verbose=False):
    
    # TODO: batch test

    results = {'n_step':np.zeros(num_episodes),
               'solved':np.zeros(num_episodes),
               'reward':np.zeros(num_episodes)}

    for i in range(num_episodes):

        obs = task.reset()
        if verbose: print('task: ', task.agent_ini_pos, '-->', task.goal_pos, end=' ...... ')
        state = torch.from_numpy(obs['image'].astype(np.float32)).unsqueeze(0).to(agent.device)
        goal = torch.from_numpy(obs['goal'].astype(np.float32)).unsqueeze(0).to(agent.device)

        for t in range(max_steps):
            action = agent.step(state=state, goal=goal)
            next_obs, reward, done, _ = task.step(action.item())
            next_state = torch.from_numpy(next_obs['image'].astype(np.float32)).unsqueeze(0).to(agent.device)

            if done:
                if verbose: print('arrived! agent_pos: ', task.env.env.env.env.agent_pos)
                next_state = None

            # Move to the next state
            state = next_state

            if done or t+1 >= max_steps:
                break

        if verbose: print('')

        results['n_step'][i] = t+1
        results['solved'][i] = done
        results['reward'][i] = reward # reward at last step

    return results


def run(run_ID,
        log_dir,
        cuda_idx,
        task_type,
        memory):    
    
    device = torch.device("cuda:%d" % cuda_idx)
    
    run_key = '{}_{}_run{}_{}'.format(task_type, 
                                      memory, 
                                      run_ID,
                                      datetime.now().strftime('%y%m%d%H%M'))
    writer = SummaryWriter(log_dir+run_key)
    
    task = FourRoomsTask(task_type=task_type, 
                         goalcond=True)
    obs = task.reset()
    if memory == 'replaybuffer':
        memory = ReplayBuffer()
        
    agent = DQN(state_dim=obs['image'].shape, 
                action_dim=4, 
                goalcond=True, 
                device=device, 
                memory=memory)
    
    train(task, agent, writer=writer, num_epochs=50)
    
    print('two thousand years later...')
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID', help='run identifier', type=int, default=0)
    parser.add_argument('--log_dir', help='logging directory', default='/data5/liyuxuan/remerge/')
    parser.add_argument('--cuda_idx', help='gpu to use', type=int, default=4)
    parser.add_argument('--task_type', help='task to run', default='ti-1')
    parser.add_argument('--memory', help='memory to use', default='replaybuffer')
    args = parser.parse_args()
    run(run_ID=args.run_ID,
        log_dir=args.log_dir,
        cuda_idx=args.cuda_idx,
        task_type=args.task_type,
        memory=args.memory)