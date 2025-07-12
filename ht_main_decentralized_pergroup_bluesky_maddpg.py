import sys
PATH_ENV = "./envs" 
sys.path.append(PATH_ENV)
import wandb
import bluesky as bs
from torch.utils.data import DataLoader, TensorDataset
import warnings
import csv
import os
import time
import random

from typing import Union
import time
import torch
print("Pytorch Verison")
print(torch.__version__)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from itertools import count
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython import display
import argparse

from ht_buffer_bluesky_maddpg import ReplayBuffer
from ht_alg_bluesky_maddpg import MADDPG
from ht_utils_bluesky_maddpg import seed_random, plot_result, save_result, schedule_epsilon
import copy

warnings.filterwarnings("ignore", category=UserWarning, message=".*?.*?")

class Runner:
    def __init__(self, args, env_name, number, seed, device):
        self.args = args
        self.device = device
        self.env_name = env_name
        self.number = number
        self.seed = seed
        self.render = 'sim' #'server'

        # Create env
        if self.env_name == "bluesky_reduced":
            from envs import my_matc_env_reduced 
            self.env = my_matc_env_reduced.MATCEnv('routes/matc/routes_two_companies_reduced.json', render_mode=self.render)
            self.args.N = 4
        else:
            from envs import my_matc_env
            self.env = my_matc_env.MATCEnv('routes/matc/routes_two_companies.json', render_mode=self.render)
            self.args.N = 20

        from envs.scenario.utils import haversine_distance, normalize
        import bluesky as bs
        
        self.args.companylist = ['company1', 'company2']
        self.args.Hetero = len(self.args.companylist)

        self.all_agents = list(self.env.dones.keys())
        self.group1_agents = [agent for agent in self.all_agents if agent.startswith('AA') or agent.startswith('AC')]
        self.group2_agents = [agent for agent in self.all_agents if agent.startswith('AB') or agent.startswith('AD')]
        self.args.group1_idx = [idx for idx, agent in enumerate(self.all_agents) if agent.startswith('AA') or agent.startswith('AC')]
        self.args.group2_idx = [idx for idx, agent in enumerate(self.all_agents) if agent.startswith('AB') or agent.startswith('AD')]

        self.args.others = 3
        self.args.each_obs_dim_n = 6
        self.args.other_dim_n = 5 
        self.args.act_dim_n = 3
        self.args.real_obs_dim_n = self.args.each_obs_dim_n + (self.args.others * self.args.other_dim_n)
        self.args.obs_dim_n = [self.args.real_obs_dim_n  for i in range(self.args.N)]  # obs dimensions of N agents
        self.args.action_dim_n = [self.args.act_dim_n for i in range(self.args.N)]
        
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_dim_n={}".format(self.args.act_dim_n))

        self.SAVE_DIR = "./fig/"
        self.SAVE_GIF_DIR = "./fig/gif/"

        seed_random(self.seed)

        if self.args.algorithm == "MADDPG":
            print("Algorithm: MADDPG")
            self.agent_n = [MADDPG(args, group_id, self.device) for group_id in range(self.args.Hetero)]
            #self.agent_n = [MADDPG(args, group_id, self.device) for group_id in range(self.args.N)]
        else:
            print("Other Algorithm")

        self.replay_buffer = ReplayBuffer(self.args, self.device)
        #self.replay_buffer_n = [ReplayBuffer(self.args, self.device) for _ in range(self.args.Hetero)]

        self.evaluate_rewards = []  # Record the rewards during the evaluation
        self.evaluate_collision = []  # Record the rewards during the evaluation
        self.records_steps = []

        self.records_r_data = []
        self.IS_GIF = False
        self.total_steps = 0
        self.episode_counter = 0

        self.noise_std = self.args.noise_std_init
        self.init_epsilon = 1.0 
        self.min_epsilon = 0.01
        self.slope_epsilon = 0.5
        if self.args.use_noise_decay:
            self.epsilon = self.init_epsilon
        else:
            self.epsilon = 0


    def run(self, ):
        _, _ = self.evaluate_policy()

        processed_states, infos = self.env.reset(seed=self.seed)
        self.all_agents = list(self.env.dones.keys())
        self.group1_agents = [agent for agent in self.all_agents if agent.startswith('AA') or agent.startswith('AC')]
        self.group2_agents = [agent for agent in self.all_agents if agent.startswith('AB') or agent.startswith('AD')]
        self.args.group1_idx = [idx for idx, agent in enumerate(self.all_agents) if agent.startswith('AA') or agent.startswith('AC')]
        self.args.group2_idx = [idx for idx, agent in enumerate(self.all_agents) if agent.startswith('AB') or agent.startswith('AD')]

        dones = {agent: 0 for agent in processed_states.keys()}  # Initialize done states for all agents
        episode_actions = {agent: [] for agent in processed_states.keys()}
        check_step = 0

        while self.total_steps < self.args.max_train_steps:
            agent_keys = list(processed_states.keys())
            agent_keys_company1 = [agent for agent in agent_keys if agent.startswith('AA') or agent.startswith('AC')]
            agent_keys_company2 = [agent for agent in agent_keys if agent.startswith('AB') or agent.startswith('AD')]

            obs = self.generate_padded_obs()
            for ownship_agent in agent_keys:
                ownship_state = np.array([value for _, value in processed_states[ownship_agent]['ownship_state'].items()], dtype=np.float32) #(6,)
                intruder_states = processed_states[ownship_agent]['intruder_states'].astype(np.float32)
                concat_intruder_states = intruder_states.flatten() #(15,)
                per_obs = np.concatenate((ownship_state, concat_intruder_states)) #(21,)
                obs[ownship_agent] = per_obs

            act = self.generate_padded_act()
            Actions = {}
            # Company #1
            for idx, ownship_agent in enumerate(agent_keys_company1):
                tensor_obs = torch.from_numpy(obs[ownship_agent]).unsqueeze(0).to(device)
                cur_act, discrete_cur_act = self.agent_n[0].choose_action(tensor_obs, noise_std=self.epsilon)
                Actions[ownship_agent] = discrete_cur_act.item()
                act[ownship_agent] = cur_act
            # Company #2
            for idx, ownship_agent in enumerate(agent_keys_company2):
                tensor_obs = torch.from_numpy(obs[ownship_agent]).unsqueeze(0).to(device)
                cur_act, discrete_cur_act = self.agent_n[1].choose_action(tensor_obs, noise_std=self.epsilon)
                Actions[ownship_agent] = discrete_cur_act.item()
                act[ownship_agent] = cur_act
            for ownship_agent in processed_states.keys():
                if dones.get(ownship_agent, False):
                    Actions.pop(ownship_agent, None)

            next_processed_states, rewards, terminations, truncations, dones = self.env.step(Actions)

            shared_reward = sum(list(rewards.values()))
            i_reward = np.zeros( self.args.Hetero, dtype=np.float32)
            for item in zip(list(rewards.keys()), list(rewards.values())):
                if item[0] in self.group1_agents:
                    i_reward[0] += item[1]
                else:
                    i_reward[1] += item[1]
            i_reward = i_reward.tolist()
            shared_reward = float(shared_reward) 
            S_R = {}
            for item in self.all_agents:
                S_R[item] = shared_reward
            I_R = self.generate_padded_rwd()
            for item in zip(list(rewards.keys()), i_reward):
                I_R[item[0]] = item[1]


            n_agent_keys = list(next_processed_states.keys())
            n_agent_keys_company1 = [agent for agent in n_agent_keys if agent.startswith('AA') or agent.startswith('AC')]
            n_agent_keys_company2 = [agent for agent in n_agent_keys if agent.startswith('AB') or agent.startswith('AD')]

            n_obs = self.generate_padded_obs()
            for ownship_agent in n_agent_keys:
                next_ownship_state = np.array([value for _, value in next_processed_states[ownship_agent]['ownship_state'].items()], dtype=np.float32) #(6,)
                next_intruder_states = next_processed_states[ownship_agent]['intruder_states'].astype(np.float32)
                next_concat_intruder_states = next_intruder_states.flatten() #(15,)
                per_next_obs = np.concatenate((next_ownship_state, next_concat_intruder_states)) #(21,)
                n_obs[ownship_agent] = per_next_obs


            np_obs = [ np.expand_dims(item, axis=0) for item in list(obs.values())] 
            np_act = []
            for item in list(act.values()):
                if torch.is_tensor(item):
                    np_act.append(item.detach().cpu().numpy())
                else:
                    np_act.append(item)
            #using shared rewards (per team)
            np_r = [ np.expand_dims(item, axis=0).astype(np.float32) for item in list(S_R.values())] 
            np_n_obs = [ np.expand_dims(item, axis=0) for item in list(n_obs.values())] 
            np_dones = [ np.expand_dims(item, axis=0) for item in list(self.env.dones.values())] 

            self.replay_buffer.store_transition(np_obs, np_act, np_r, np_n_obs, np_dones)
            processed_states = next_processed_states
            check_step += 1
            self.total_steps += 1

            if self.args.use_noise_decay:
                self.epsilon = schedule_epsilon(self.init_epsilon, self.min_epsilon, self.slope_epsilon * self.args.max_train_steps, self.total_steps)

            if self.replay_buffer.current_size > self.args.batch_size:
                # for agent_id in self.all_agents:
                #     if agent_id in self.group1_agents:
                #         self.agent_n[0].train(self.replay_buffer, self.agent_n[0])
                #     else:
                #         self.agent_n[1].train(self.replay_buffer, self.agent_n[1])

                #fully decentralized (no sharing)
                # for agent_id in range(self.args.N):
                #     self.agent_n[agent_id].train(self.replay_buffer, self.agent_n)

                #decentralized but param sharing
                for group_id in range(self.args.Hetero):
                    self.agent_n[group_id].train_group(self.replay_buffer, self.agent_n)

            if self.total_steps % self.args.evaluate_freq == 0:
                p1, p2 = self.evaluate_policy()
                self.evaluate_collision.append(p1)
                self.evaluate_rewards.append(p2)
                self.records_steps.append(self.total_steps) 
                plot_result(self.records_steps, self.evaluate_collision, self.args.Hetero, "C", self.SAVE_DIR, self.env_name)
                plot_result(self.records_steps, self.evaluate_rewards, self.args.Hetero, "R", self.SAVE_DIR, self.env_name)

            if (all(self.env.dones.values())) or (check_step >= self.args.max_limit_steps):
                print("############# During {} training steps: End of Episode with total steps {} #############".format(self.total_steps, check_step))
                processed_states, infos = self.env.reset(seed=self.seed)
                self.all_agents = list(self.env.dones.keys())
                self.group1_agents = [agent for agent in self.all_agents if agent.startswith('AA') or agent.startswith('AC')]
                self.group2_agents = [agent for agent in self.all_agents if agent.startswith('AB') or agent.startswith('AD')]
                dones = {agent: 0 for agent in processed_states.keys()}  # Initialize done states for all agents
                episode_actions = {agent: [] for agent in processed_states.keys()}
                check_step = 0

        plot_result(self.records_steps, self.evaluate_collision, self.args.Hetero, "C", self.SAVE_DIR, self.env_name)
        plot_result(self.records_steps, self.evaluate_rewards, self.args.Hetero, "R", self.SAVE_DIR, self.env_name)

        save_result(self.records_steps, self.evaluate_collision, self.args.Hetero, self.env_name, self.args.algorithm, "C", self.SAVE_DIR+"results_data/")
        save_result(self.records_steps, self.evaluate_rewards, self.args.Hetero, self.env_name, self.args.algorithm, "R", self.SAVE_DIR+"results_data/")

        print("DONE")

    def generate_padded_obs(self,):
        gen_obs = {}
        total_size = self.args.each_obs_dim_n + (self.args.others * self.args.other_dim_n)
        padding = np.zeros(total_size, dtype=np.float32)
        for agent_id in self.all_agents:
            gen_obs[agent_id] = padding
        return gen_obs
    
    def generate_padded_act(self,):
        gen_act = {}
        total_size = self.args.act_dim_n 
        padding = np.zeros(total_size, dtype=np.float32)
        padding = np.expand_dims(padding, axis=0)
        for agent_id in self.all_agents:
            gen_act[agent_id] = padding
        return gen_act
    
    def generate_padded_rwd(self,):
        gen_rwd = {}
        for agent_id in self.all_agents:
            gen_rwd[agent_id] = 0
        return gen_rwd
    
    def generate_onehot(self,dim,idx):
        onehot_vec = np.zeros(dim)
        onehot_vec[idx] = 1
        return onehot_vec

    @torch.no_grad()
    def evaluate_policy(self, ):
        for ag in self.agent_n:
            ag.actor.eval()
            ag.critic.eval()
            ag.actor_target.eval()
            ag.critic_target.eval()

        performance_collision = []
        performance_rwd = [] 
        evaluate_reward = np.zeros(self.args.Hetero) #[0] * self.args.N #0
        eval_rs = []
        for EVAL_TIMES in range(self.args.evaluate_times):

            processed_states, infos = self.env.reset(seed=self.seed)
            self.all_agents = list(self.env.dones.keys())
            self.group1_agents = [agent for agent in self.all_agents if agent.startswith('AA') or agent.startswith('AC')]
            self.group2_agents = [agent for agent in self.all_agents if agent.startswith('AB') or agent.startswith('AD')]
            dones = {agent: 0 for agent in processed_states.keys()}
            episode_actions = {agent: [] for agent in processed_states.keys()}

            episode_reward = np.zeros(self.args.Hetero)
            frame_list = []
            DONE = False
            rwd = self.generate_padded_rwd()
            steps = 0
            while not DONE:
                steps += 1
                agent_keys = list(processed_states.keys())
                agent_keys_company1 = [agent for agent in agent_keys if agent.startswith('AA') or agent.startswith('AC')]
                agent_keys_company2 = [agent for agent in agent_keys if agent.startswith('AB') or agent.startswith('AD')]

                obs = {}
                obs = self.generate_padded_obs()
                for ownship_agent in agent_keys:
                    ownship_state = np.array([value for _, value in processed_states[ownship_agent]['ownship_state'].items()], dtype=np.float32) #(6,)
                    intruder_states = processed_states[ownship_agent]['intruder_states'].astype(np.float32)
                    concat_intruder_states = intruder_states.flatten() #(15,)
                    per_obs = np.concatenate((ownship_state, concat_intruder_states)) #(21,)
                    obs[ownship_agent] = per_obs

                act = self.generate_padded_act()
                Actions = {}
                # Company 1
                for idx, ownship_agent in enumerate(agent_keys_company1):
                    tensor_obs = torch.from_numpy(obs[ownship_agent]).unsqueeze(0).to(device)
                    cur_act, discrete_cur_act = self.agent_n[0].choose_action(tensor_obs, noise_std=0)
                    Actions[ownship_agent] = discrete_cur_act.item()
                    act[ownship_agent] = cur_act
                # Company 2
                for idx, ownship_agent in enumerate(agent_keys_company2):
                    tensor_obs = torch.from_numpy(obs[ownship_agent]).unsqueeze(0).to(device)
                    cur_act, discrete_cur_act = self.agent_n[1].choose_action(tensor_obs, noise_std=0)
                    Actions[ownship_agent] = discrete_cur_act.item()
                    act[ownship_agent] = cur_act
                for ownship_agent in processed_states.keys():
                    if dones.get(ownship_agent, False):
                        Actions.pop(ownship_agent, None)

                next_processed_states, rewards, terminations, truncations, dones = self.env.step(Actions)

                n_agent_keys = list(next_processed_states.keys())
                n_agent_keys_company1 = [agent for agent in n_agent_keys if agent.startswith('AA') or agent.startswith('AC')]
                n_agent_keys_company2 = [agent for agent in n_agent_keys if agent.startswith('AB') or agent.startswith('AD')]

                n_obs = self.generate_padded_obs()
                for ownship_agent in n_agent_keys:
                    next_ownship_state = np.array([value for _, value in next_processed_states[ownship_agent]['ownship_state'].items()], dtype=np.float32) #(6,)
                    next_intruder_states = next_processed_states[ownship_agent]['intruder_states'].astype(np.float32)
                    next_concat_intruder_states = next_intruder_states.flatten() #(15,)
                    per_next_obs = np.concatenate((next_ownship_state, next_concat_intruder_states)) #(21,)
                    n_obs[ownship_agent] = per_next_obs

                if len(list(rewards.keys())) != 0:
                    for agent in list(rewards.keys()):
                        rwd[agent] += rewards[agent]

                processed_states = next_processed_states

                if (all(self.env.dones.values())) or (steps >= self.args.max_limit_steps):
                    print("EVAL iters {}: End of Episode with total steps {}".format(EVAL_TIMES+1, steps))
                    DONE = True
            
            #calculate - performance collision
            collided_agents = self.env.collided_agents
            g_col, g1_col, g2_col = 0, 0, 0
            if len(collided_agents) != 0:
                for item in collided_agents:
                    if item[0] in self.group1_agents and item[1] in self.group1_agents:
                        g1_col += 1
                    elif item[0] in self.group2_agents and item[1] in self.group2_agents:
                        g2_col += 1
                    else:
                        g_col += 1
                performance_collision.append((g_col+g1_col,g_col+g2_col))
            else:
                performance_collision.append((g_col+g1_col,g_col+g2_col))
            assert len(self.env.collided_agents) == (g_col + g1_col + g2_col)

            #calculate - performance rewards
            rwd_agent_list = list(rwd.keys())
            rwd_cum_list = list(rwd.values())
            score_all, score_g1, score_g2 = 0, 0, 0
            for item in zip(rwd_agent_list, rwd_cum_list):
                score_all += item[1]
                if item[0] in self.group1_agents:
                    score_g1 += item[1]
                else:
                    score_g2 += item[1]
            performance_rwd.append((score_all/steps, score_g1/steps, score_g2/steps))

        group1_col, group2_col = zip(*performance_collision)
        np_group1_col = np.array(group1_col)
        np_group2_col = np.array(group2_col)
        p_c = [(np.mean(np_group1_col), np.std(np_group1_col)), (np.mean(np_group2_col), np.std(np_group2_col))]

        groupall_rwd, group1_rwd, group2_rwd = zip(*performance_rwd)
        np_groupall_rwd = np.array(groupall_rwd)
        np_group1_rwd = np.array(group1_rwd)
        np_group2_rwd = np.array(group2_rwd)
        p_r = [(np.mean(np_group1_rwd), np.std(np_group1_rwd)), (np.mean(np_group2_rwd), np.std(np_group2_rwd)), (np.mean(np_groupall_rwd), np.std(np_groupall_rwd))]

        for ag in self.agent_n:
            ag.actor.train()
            ag.critic.train()
            ag.actor_target.train()
            ag.critic_target.train()

        return p_c, p_r 


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG in the customized bluesky environment")
    parser.add_argument("--max_train_steps", type=int, default=int(5000), help="Maximum number of training steps") #1000 #20000
    parser.add_argument("--max_limit_steps", type=int, default=20000, help="Bluesky max step limit") #1e6
    parser.add_argument("--evaluate_times", type=int, default=5, help="Evaluate times")
    parser.add_argument("--evaluate_freq", type=int, default=100, help="Evaluate the policy every freq steps")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to use the noise")
    parser.add_argument("--gumbel_type", type=bool, default=True, help="onehot gumbel softmax(hard:True)")

    parser.add_argument("--algorithm", type=str, default="MADDPG", help="MADDPG")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="Capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden nodes in the neural network") 
    parser.add_argument("--lr_a", type=float, default=5e-4, help="Learning rate - actor")
    parser.add_argument("--lr_c", type=float, default=5e-4, help="Learning rate - critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update of the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")

    parser.add_argument("--test_max_episodes", type=int, default=int(1), help="Training episodes") #1e6
    parser.add_argument("--max_action", type=float, default=0.5, help="Maximum action") #not used

    parser.add_argument("--env_nums", type=int, default=1, help="parallel environments for training")
    parser.add_argument("--env_continuous_actions", type=bool, default=True, help="is continuous action") #not used
    parser.add_argument("--env_term_trun_split", type=bool, default=True, help="Dones") #not used 
    parser.add_argument("--env_max_steps", type=int, default=300, help="Maximum episode steps") #300
    parser.add_argument("--env_save_gif", type=int, default=50000, help="Save gif or not") #50000

    parser.add_argument("--noise_std_init", type=float, default=0.2, help="Gaussian noise init")
    parser.add_argument("--noise_std_min", type=float, default=0.05, help="Gaussian noise min")
    parser.add_argument("--noise_decay_steps", type=float, default=3e5, help="Steps to the noise minimum")

    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    print('Using PyTorch version:', torch.__version__)
    if torch.cuda.is_available():
        print('Using GPU, device name:', torch.cuda.get_device_name(0))
        device = torch.device('cuda')
    else:
        print('No GPU, using CPU instead.') 
        device = torch.device('cpu')

    env_names = ["bluesky", "bluesky_reduced"]
    play_env = env_names[0]
    runner = Runner(args, env_name=play_env, number=1, seed=42, device=device)
    runner.run()
    #runner.test_env()

    print("DONE")
