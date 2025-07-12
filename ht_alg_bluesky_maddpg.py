import torch
import torch.nn.functional as F
import numpy as np
import copy
from ht_networks_bluesky_maddpg import Actor, Critic_MADDPG
import random


class MADDPG(object):
    def __init__(self, args, agent_id, device):
        self.N = args.N
        self.args = args
        self.device = device
        self.agent_id = agent_id
        self.max_action = args.max_action
        self.action_dim = args.action_dim_n[agent_id]
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.tau = args.tau
        self.use_grad_clip = args.use_grad_clip
        self.actor = Actor(args, agent_id).to(self.device)
        self.critic = Critic_MADDPG(args).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def choose_action(self, obs, noise_std):

        # a = self.actor(obs)
        # dist = torch.distributions.Categorical(a)
        # discrete_a = dist.sample()
        
        if np.random.rand() < noise_std:
            #random
            onehot_a = np.zeros(self.action_dim)
            rnd_a = random.randrange(self.action_dim)
            onehot_a[rnd_a] = 1
            a = torch.tensor([onehot_a], device=self.device, dtype=torch.float)
            discrete_a = torch.tensor([rnd_a], device=self.device) #dtype=torch.long
        else:
            a = self.actor(obs)
            dist = torch.distributions.Categorical(a)
            discrete_a = dist.sample()

        return a, discrete_a

    def train(self, replay_buffer, agent_n):
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

        with torch.no_grad():  
            batch_a_next_n = [agent.actor_target(batch_obs_next) for agent, batch_obs_next in zip(agent_n, batch_obs_next_n)]
            Q_next = self.critic_target(batch_obs_next_n, batch_a_next_n)
            target_Q = batch_r_n[self.agent_id] + self.gamma * (1 - batch_done_n[self.agent_id]) * Q_next  # shape:(batch_size,1)

        current_Q = self.critic(batch_obs_n, batch_a_n) 
        critic_loss = F.mse_loss(target_Q, current_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        batch_a_n[self.agent_id] = self.actor(batch_obs_n[self.agent_id])
        actor_loss = -self.critic(batch_obs_n, batch_a_n).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        # soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def train_group(self, replay_buffer, agent_n):
        #agent: designated group
        #agent_n: list, all groups(agents)
        if self.agent_id == 0:
            group_agent_idx = self.args.group1_idx
        else:
            group_agent_idx = self.args.group2_idx

        for agent_idx in group_agent_idx:

            batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

            with torch.no_grad():
                batch_a_next_n = []
                for agent_idxidx, batch_obs_next in zip(list(range(self.args.N)), batch_obs_next_n):
                    if agent_idxidx in self.args.group1_idx:
                        batch_a_next_n.append(agent_n[0].actor_target(batch_obs_next))
                    else:
                        batch_a_next_n.append(agent_n[1].actor_target(batch_obs_next))
                Q_next = self.critic_target(batch_obs_next_n, batch_a_next_n)
                target_Q = batch_r_n[agent_idx] + self.gamma * (1 - batch_done_n[agent_idx]) * Q_next

            current_Q = self.critic(batch_obs_n, batch_a_n)
            critic_loss = F.mse_loss(target_Q, current_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.critic_optimizer.step()

            batch_a_n[agent_idx] = self.actor(batch_obs_n[agent_idx])
            actor_loss = -self.critic(batch_obs_n, batch_a_n).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, env_name, algorithm, number, total_steps, agent_id):
        torch.save(self.actor.state_dict(), "./fig/{}_actor_number_{}_step_{}_agent_{}.pth".format(algorithm, number, int(total_steps), agent_id))