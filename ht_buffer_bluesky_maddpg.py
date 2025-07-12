import torch
import numpy as np

class ReplayBuffer(object):
    def __init__(self, args, device):
        self.N = args.N  # 1 per single group
        self.args = args
        self.device = device
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.count = 0
        self.current_size = 0
        self.buffer_obs_n, self.buffer_a_n, self.buffer_r_n, self.buffer_s_next_n, self.buffer_done_n = [], [], [], [], []
        for agent_id in range(self.N):
            self.buffer_obs_n.append(np.empty((self.buffer_size, args.real_obs_dim_n)))
            self.buffer_a_n.append(np.empty((self.buffer_size, args.act_dim_n)))
            self.buffer_r_n.append(np.empty((self.buffer_size, 1)))
            self.buffer_s_next_n.append(np.empty((self.buffer_size, args.real_obs_dim_n)))
            self.buffer_done_n.append(np.empty((self.buffer_size, 1)))

    def store_transition(self, obs_n, a_n, r_n, obs_next_n, done_n):
        for agent_id in range(self.N):
            self.buffer_obs_n[agent_id][self.count] = obs_n[agent_id]
            self.buffer_a_n[agent_id][self.count] = a_n[agent_id]
            self.buffer_r_n[agent_id][self.count] = r_n[agent_id]
            self.buffer_s_next_n[agent_id][self.count] = obs_next_n[agent_id]
            self.buffer_done_n[agent_id][self.count] = done_n[agent_id]
        self.count = (self.count + 1) % self.buffer_size  
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, ):
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = [], [], [], [], []
        for agent_id in range(self.N):
            batch_obs_n.append(torch.tensor(self.buffer_obs_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_a_n.append(torch.tensor(self.buffer_a_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_r_n.append(torch.tensor(self.buffer_r_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_obs_next_n.append(torch.tensor(self.buffer_s_next_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_done_n.append(torch.tensor(self.buffer_done_n[agent_id][index], dtype=torch.float).to(self.device))
        return batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n

    # def sample_w_checker(self, group_idxs): #Not Valid
    #     valid_agent_idx = group_idxs

    #     index = np.random.choice(self.current_size, size=self.batch_size, replace=False) #self.batch_size
    #     batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = [], [], [], [], []

    #     for agent_id in range(self.N):
    #         batch_obs_n.append(torch.tensor(self.buffer_obs_n[agent_id][index], dtype=torch.float).to(self.device))
    #         batch_a_n.append(torch.tensor(self.buffer_a_n[agent_id][index], dtype=torch.float).to(self.device))
    #         batch_r_n.append(torch.tensor(self.buffer_r_n[agent_id][index], dtype=torch.float).to(self.device))
    #         batch_obs_next_n.append(torch.tensor(self.buffer_s_next_n[agent_id][index], dtype=torch.float).to(self.device))
    #         batch_done_n.append(torch.tensor(self.buffer_done_n[agent_id][index], dtype=torch.float).to(self.device))

    #     agent_checker = []
    #     for agent_id in range(self.N):
    #         per_agent_checker = []
    #         for item in self.buffer_a_n[agent_id][index].tolist():
    #             if np.array_equal(item, np.zeros(self.args.act_dim_n, dtype=np.float32)):
    #                 per_agent_checker.append(0)
    #             else:
    #                 if agent_id in self.args.group1_idx:
    #                     per_agent_checker.append(1)
    #                 else:
    #                     per_agent_checker.append(2)
    #         agent_checker.append(per_agent_checker)

    #     np_agent_checker = np.array(agent_checker)
    #     T_agent_checker = np_agent_checker.T.tolist()

    #     return batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n, T_agent_checker