import wandb
import numpy as np
from envs import my_matc_env
from PPO_attention import PPO_Agent
from PPO_attention import Memory
from PPO_attention import compute_gae, discount
from PPO_attention import schedule_beta
import bluesky as bs
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import warnings
from envs.scenario.utils import haversine_distance, normalize
import csv
import os
import time

warnings.filterwarnings("ignore", category=UserWarning, message=".*?.*?")

MAX_NUM_AGENTS = 20
NUM_INTRUDERS = 3

STATE_SIZE = 6
INTRUDER_STATE_SIZE = 5
ACTION_SIZE = 3

NUM_EPISODES = 10000
BATCH_SIZE = 512
EPOCHS = 10
ALPHA = 0.2
BETA = 1e-3
INITIAL_BETA = 0.01
FINAL_BETA = 0.001
LR = 1e-4
POLICY_UPDATE_TIME_STEP = 1024
SHUFFLE = True

TRAINING = True
WANDB = False
RENDER_MODE = 'sim'
# Model name
MODEL_NAME_COMPANY1 = 'best_policy_company1_june25.pt'
MODEL_NAME_COMPANY2 = 'best_policy_company2_june25.pt'

MAX_STEPS_SO_FAR = 20000


# Example usage
if __name__ == '__main__':

    # Initialize the PPO agent
    state_size = STATE_SIZE 
    context_size = INTRUDER_STATE_SIZE
    action_size = ACTION_SIZE  
    num_intruders = NUM_INTRUDERS  
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    num_agents = MAX_NUM_AGENTS
    numEpisodes = NUM_EPISODES

    # Initializing the environment
    env = my_matc_env.MATCEnv('routes/matc/routes_two_companies.json', render_mode=RENDER_MODE)

    # Company list
    companylist = ['company1', 'company2']

    # Initialize the PPO agent
    company1_agent = PPO_Agent(state_size=state_size, context_size=context_size, action_size=action_size, num_intruders=num_intruders)
    company2_agent = PPO_Agent(state_size=state_size, context_size=context_size, action_size=action_size, num_intruders=num_intruders)

    device = company1_agent.device

    if not TRAINING:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Load the best model
        company1_agent.load(name=f'{script_dir}/trained_models/best_policy_company1_june9.pt')
        company2_agent.load(name=f'{script_dir}/trained_models/best_policy_company2_june9.pt')

    # Initialize the memory
    memory_company1 = Memory()
    memory_company2 = Memory()

    # Initialize wandb
    if WANDB:
        wandb.init(project="atc_marl_two_company", name="Initial-training-May-30", config={
            "state_size": state_size,
            "action_size": action_size,
            "num_intruders": num_intruders,
            "num_agents": num_agents,
            "numEpisodes": numEpisodes,
            "batch_size": batch_size,
            "epochs": epochs
        })

    if TRAINING:  
        best_avg_rewards_company1 = float('-inf')  # Initialize the best average rewards
        best_avg_rewards_company2 = float('-inf')  # Initialize the best average rewards

    # Training loop
    for episode in range(numEpisodes):

        # Starting time for the episode
        start_time = time.time()

        rewards = {}
        processed_states, infos = env.reset(seed=42)
        dones = {agent: 0 for agent in processed_states.keys()}  # Initialize done states for all agents
        episode_actions = {agent: [] for agent in processed_states.keys()}
        # for i in range(episode_length):
        counter = 0
        episodic_losses = []
        entropies = []
        kl_divergences = []
        training_steps = 0
        collided_agents_info = {}

        # update beta 
        beta_company1 = schedule_beta(episode=episode, decay_episode=1000, initial_beta=INITIAL_BETA, final_beta=FINAL_BETA)
        beta_company2 = schedule_beta(episode=episode, decay_episode=1000, initial_beta=INITIAL_BETA, final_beta=FINAL_BETA)

        # print(f"Beta: {beta}")
        END_EPISODE = False
        while not END_EPISODE: 

            company1_agent.model.to(torch.device('cpu'))
            company2_agent.model.to(torch.device('cpu'))

            Actions = {}
            LogProbs = {}

            # Prepare batched inputs for all agents
            agent_keys = list(processed_states.keys())

            # if len(agent_keys) == 0:
            #     print("No agents left in the environment. Ending episode.")
            #     break

            # extract company 1 and company 2 agents
            agent_keys_company1 = [agent for agent in agent_keys if agent.startswith('AA') or agent.startswith('AC')]
            agent_keys_company2 = [agent for agent in agent_keys if agent.startswith('AB') or agent.startswith('AD')]

            # Collect all states and contexts as lists (CPU-side)
            all_states_company1 = []
            all_states_company2 = []
            all_contexts_company1 = []
            all_contexts_company2 = []

            for ownship_agent in agent_keys:
                ownship_state = np.array([value for _, value in processed_states[ownship_agent]['ownship_state'].items()], dtype=np.float32)
                intruder_states = processed_states[ownship_agent]['intruder_states'].astype(np.float32)
                # if ownship_agent == 'AD0003':
                #     print(f"Ownship Agent: {ownship_agent}, State: {ownship_state}, Intruder States: {intruder_states}")

                if ownship_agent in agent_keys_company1:
                    all_states_company1.append(ownship_state)
                    all_contexts_company1.append(intruder_states)
                elif ownship_agent in agent_keys_company2:
                    all_states_company2.append(ownship_state)
                    all_contexts_company2.append(intruder_states)

            # Ensure all_states and all_contexts have the correct shape
            # Company #1
            if len(all_states_company1) == 0:
                batched_states_company1 = []
                batched_contexts_company1 = []
            elif len(all_states_company1) == 1:
                batched_states_company1 = torch.tensor(all_states_company1[0], dtype=torch.float32).unsqueeze(0)
                batched_contexts_company1 = torch.tensor(all_contexts_company1[0], dtype=torch.float32).unsqueeze(0)
            else:
                batched_states_company1 = torch.tensor(np.vstack(all_states_company1), dtype=torch.float32)
                batched_contexts_company1 = torch.tensor(np.stack(all_contexts_company1), dtype=torch.float32)

            # Company #2
            if len(all_states_company2) == 0:
                batched_states_company2 = []
                batched_contexts_company2 = []
            elif len(all_states_company2) == 1:
                batched_states_company2 = torch.tensor(all_states_company2[0], dtype=torch.float32).unsqueeze(0)
                batched_contexts_company2 = torch.tensor(all_contexts_company2[0], dtype=torch.float32).unsqueeze(0)
            else:
                batched_states_company2 = torch.tensor(np.vstack(all_states_company2), dtype=torch.float32)
                batched_contexts_company2 = torch.tensor(np.stack(all_contexts_company2), dtype=torch.float32)

            # Get action probabilities for all agents in a single call
            if TRAINING:
                with torch.no_grad():
                    # Company #1
                    if len(batched_states_company1) > 0:
                        policies_company1, old_values_company1 = company1_agent.model(batched_states_company1, batched_contexts_company1)
                        dists_company1 = torch.distributions.Categorical(policies_company1)
                        actions_company1 = dists_company1.sample()
                        logprobs_company1 = dists_company1.log_prob(actions_company1)
                    # Company #2
                    if len(batched_states_company2) > 0:
                        policies_company2, old_values_company2 = company2_agent.model(batched_states_company2, batched_contexts_company2)
                        dists_company2 = torch.distributions.Categorical(policies_company2)
                        actions_company2 = dists_company2.sample()
                        logprobs_company2 = dists_company2.log_prob(actions_company2)
            else: # Take the action with the highest probability
                with torch.no_grad():
                    # Company #1
                    if len(batched_states_company1) > 0:
                        policies_company1, old_values_company1 = company1_agent.model(batched_states_company1, batched_contexts_company1)
                        actions_company1 = torch.argmax(policies_company1, dim=1)
                        logprobs_company1 = torch.zeros_like(actions_company1, dtype=torch.float32)
                    # Company #2
                    if len(batched_states_company2) > 0:
                        policies_company2, old_values_company2 = company2_agent.model(batched_states_company2, batched_contexts_company2)
                        actions_company2 = torch.argmax(policies_company2, dim=1)
                        logprobs_company2 = torch.zeros_like(actions_company2, dtype=torch.float32)

            # Assign actions, log probabilities, and old values back to individual agents
            # Company #1
            old_values_dict_company1 = {}
            for idx, ownship_agent in enumerate(agent_keys_company1):
                Actions[ownship_agent] = actions_company1[idx].item()
                LogProbs[ownship_agent] = logprobs_company1[idx].detach()
                old_values_dict_company1[ownship_agent] = old_values_company1[idx].detach()
            # Company #2
            old_values_dict_company2 = {}
            for idx, ownship_agent in enumerate(agent_keys_company2):
                Actions[ownship_agent] = actions_company2[idx].item()
                LogProbs[ownship_agent] = logprobs_company2[idx].detach()
                old_values_dict_company2[ownship_agent] = old_values_company2[idx].detach()

            for ownship_agent in processed_states.keys():
                if dones.get(ownship_agent, False):
                    # drop the agent from Actions, LogProbs, and old_values_dict
                    Actions.pop(ownship_agent, None)
                    LogProbs.pop(ownship_agent, None)
                    old_values_dict_company1.pop(ownship_agent, None)
                    old_values_dict_company2.pop(ownship_agent, None)

            # Step the environment
            next_processed_states, rewards, terminations, truncations, dones = env.step(Actions)
            # print(f"Rewards: {rewards}")
            # print(all(env.dones.values()))

            # Check for collisions
            collided_agents = env.collided_agents
            num_of_collisions = len(env.collided_agents)

            # === [5] Store experience into memory efficiently ===
            # Company #1
            for idx, ownship_agent in enumerate(agent_keys_company1):
                if ownship_agent in Actions.keys():
                    reward = rewards[ownship_agent]
                    done = dones.get(ownship_agent, 0)
                    # if reward <= -1 or reward > 0:
                    #     done = 1
                    #     dones[ownship_agent] = 1  # Ensure done is set to 1 if the agent has collided

                    # Use precomputed tensors (slices from the batched input)
                    state = batched_states_company1[idx].unsqueeze(0).detach()
                    context = batched_contexts_company1[idx].unsqueeze(0).detach()
                    action = Actions[ownship_agent]
                    logprob = LogProbs[ownship_agent].detach()
                    old_value = old_values_dict_company1[ownship_agent].detach()

                    # Store into memory
                    if reward is not None:
                        memory_company1.add(ownship_agent, state, context, action, reward, done, logprob, old_value)
            # Company #2
            for idx, ownship_agent in enumerate(agent_keys_company2):
                if ownship_agent in Actions.keys():
                    reward = rewards[ownship_agent]
                    done = dones.get(ownship_agent, 0)
                    # if reward <= -1 or reward > 0:
                    #     done = 1
                    #     dones[ownship_agent] = 1  # Ensure done is set to 1 if the agent has collided

                    # Use precomputed tensors (slices from the batched input)
                    state = batched_states_company2[idx].unsqueeze(0).detach()
                    context = batched_contexts_company2[idx].unsqueeze(0).detach()
                    action = Actions[ownship_agent]
                    logprob = LogProbs[ownship_agent].detach()
                    old_value = old_values_dict_company2[ownship_agent].detach()

                    # Store into memory
                    if reward is not None:
                        memory_company2.add(ownship_agent, state, context, action, reward, done, logprob, old_value)

            for agent in processed_states.keys():
                if agent in episode_actions.keys():
                    if agent in Actions.keys():
                        episode_actions[agent].append(Actions[agent])
                else:
                    if agent in Actions.keys():
                        episode_actions[agent] = [Actions[agent]]

            # Update the observations
            processed_states = next_processed_states.copy()

            # Calculate the total length of all states in memory
            # Increment the counter
            counter += 1

            # collided_agent_list = [item for tup in env.collided_agents for item in tup]

            if TRAINING and (all(env.dones.values()) or counter >= MAX_STEPS_SO_FAR):

                # print("Training the PPO agent...")

                # Move the models to the device
                company1_agent.model.to(device)
                company2_agent.model.to(device)
                
                # print("Training the PPO agent...")
                # Training the PPO agent
                # Check if all agents are done

                for company in companylist:
                    if company == 'company1':
                        memory = memory_company1
                        beta = beta_company1
                        # agent data
                        agent_data = memory.agent_data
                        all_states, all_contexts, all_actions, all_rewards, all_dones = [], [], [], [], []
                        all_advantages, all_returns, all_logprobs = [], [], []

                        # Company #1 Data Preparation
                        for agent in agent_data.keys():
                            # Extract data for the current agent
                            agent_states = torch.stack(agent_data[agent]['states']).to(device)
                            agent_contexts = torch.stack(agent_data[agent]['contexts']).to(device)
                            agent_actions = torch.tensor(agent_data[agent]['actions'], dtype=torch.long, device=device)
                            agent_rewards = torch.tensor(agent_data[agent]['rewards'], dtype=torch.float32, device=device).reshape(-1, 1)
                            agent_dones = torch.tensor(agent_data[agent]['dones'], dtype=torch.float32, device=device).reshape(-1, 1)
                            agent_logprobs = torch.tensor(agent_data[agent]['log_probs'], dtype=torch.float32, device=device).reshape(-1, 1)
                            agent_old_values = torch.stack(agent_data[agent]['old_values']).to(device)  # Use stored old values

                            # Compute GAE and returns for the current agent
                            advantages, returns = compute_gae(agent_rewards, agent_old_values, agent_dones)

                            # Append the current agent's data to the combined lists
                            all_states.append(agent_states)
                            all_contexts.append(agent_contexts)
                            all_actions.append(agent_actions)
                            all_rewards.append(agent_rewards)
                            all_dones.append(agent_dones)
                            all_advantages.append(advantages)
                            all_returns.append(returns)
                            all_logprobs.append(agent_logprobs)                

                        states = torch.cat(all_states, dim=0).to(device)
                        contexts = torch.cat(all_contexts, dim=0).to(device)
                        actions = torch.cat(all_actions, dim=0).to(device)
                        rewards = torch.cat(all_rewards, dim=0).to(device)
                        dones_ = torch.cat(all_dones, dim=0).to(device)
                        advantages = torch.cat(all_advantages, dim=0).to(device)
                        returns = torch.cat(all_returns, dim=0).to(device)
                        logprobs = torch.cat(all_logprobs, dim=0).to(device)

                        # Normalize advantages for the current agent
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                        # Create dataloader for minibatch training
                        dataset = TensorDataset(states, contexts, actions, logprobs, returns, advantages)
                        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

                        # PPO training loop
                        for _ in range(EPOCHS):
                            for s_b, c_b, a_b, old_lp_b, ret_b, adv_b in loader:
                                company1_agent.optimizer.zero_grad()
                                s_b = s_b.view(-1, STATE_SIZE)
                                c_b = c_b.view(-1, NUM_INTRUDERS, INTRUDER_STATE_SIZE)

                                new_policies, new_values = company1_agent.model(s_b, c_b)
                                dist = torch.distributions.Categorical(new_policies)

                                old_lp_b = old_lp_b.view(-1)
                                new_logprobs = dist.log_prob(a_b).view(-1)

                                # PPO ratio and losses
                                ratio = (new_logprobs - old_lp_b).exp()
                                surr1 = ratio * adv_b.view(-1)
                                surr2 = torch.clamp(ratio, 1 - ALPHA, 1 + ALPHA) * adv_b.view(-1)
                                policy_loss = -torch.min(surr1, surr2).mean()

                                value_loss = F.mse_loss(new_values, ret_b)
                                entropy = dist.entropy().mean()

                                loss = policy_loss + 0.5 * value_loss - beta * entropy
                                loss.backward() 
                                company1_agent.optimizer.step()
                                training_steps += 1
                                episodic_losses.append(loss.item())

                                entropies.append(entropy.item())
                                kl_divergences.append((old_lp_b - new_logprobs).mean().item())

                        # Compute avg rewards for company 1
                        avg_rewards_all_agents_company1 = float(sum(rewards)/len(rewards)) * 1000

                        # Reset the memory for the next episode
                        memory.reset()

                    elif company == 'company2':
                        memory = memory_company2
                        beta = beta_company2
                        # agent data
                        agent_data = memory.agent_data
                        all_states, all_contexts, all_actions, all_rewards, all_dones = [], [], [], [], []
                        all_advantages, all_returns, all_logprobs = [], [], []

                        # Company #2 Data Preparation
                        for agent in agent_data.keys():
                            # Extract data for the current agent
                            agent_states = torch.stack(agent_data[agent]['states']).to(device)
                            agent_contexts = torch.stack(agent_data[agent]['contexts']).to(device)
                            agent_actions = torch.tensor(agent_data[agent]['actions'], dtype=torch.long, device=device)
                            agent_rewards = torch.tensor(agent_data[agent]['rewards'], dtype=torch.float32, device=device).reshape(-1, 1)
                            agent_dones = torch.tensor(agent_data[agent]['dones'], dtype=torch.float32, device=device).reshape(-1, 1)
                            agent_logprobs = torch.tensor(agent_data[agent]['log_probs'], dtype=torch.float32, device=device).reshape(-1, 1)
                            agent_old_values = torch.stack(agent_data[agent]['old_values']).to(device)  # Use stored old values

                            # Compute GAE and returns for the current agent
                            advantages, returns = compute_gae(agent_rewards, agent_old_values, agent_dones)

                            # Append the current agent's data to the combined lists
                            all_states.append(agent_states)
                            all_contexts.append(agent_contexts)
                            all_actions.append(agent_actions)
                            all_rewards.append(agent_rewards)
                            all_dones.append(agent_dones)
                            all_advantages.append(advantages)
                            all_returns.append(returns)
                            all_logprobs.append(agent_logprobs)                

                        states = torch.cat(all_states, dim=0).to(device)
                        contexts = torch.cat(all_contexts, dim=0).to(device)
                        actions = torch.cat(all_actions, dim=0).to(device)
                        rewards = torch.cat(all_rewards, dim=0).to(device)
                        dones_ = torch.cat(all_dones, dim=0).to(device)
                        advantages = torch.cat(all_advantages, dim=0).to(device)
                        returns = torch.cat(all_returns, dim=0).to(device)
                        logprobs = torch.cat(all_logprobs, dim=0).to(device)

                        # Normalize advantages for the current agent
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                        # Create dataloader for minibatch training
                        dataset = TensorDataset(states, contexts, actions, logprobs, returns, advantages)
                        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

                        # PPO training loop
                        for _ in range(EPOCHS):
                            for s_b, c_b, a_b, old_lp_b, ret_b, adv_b in loader:
                                company2_agent.optimizer.zero_grad()
                                s_b = s_b.view(-1, STATE_SIZE)
                                c_b = c_b.view(-1, NUM_INTRUDERS, INTRUDER_STATE_SIZE)

                                new_policies, new_values = company2_agent.model(s_b, c_b)
                                dist = torch.distributions.Categorical(new_policies)

                                old_lp_b = old_lp_b.view(-1)
                                new_logprobs = dist.log_prob(a_b).view(-1)

                                # PPO ratio and losses
                                ratio = (new_logprobs - old_lp_b).exp()
                                surr1 = ratio * adv_b.view(-1)
                                surr2 = torch.clamp(ratio, 1 - ALPHA, 1 + ALPHA) * adv_b.view(-1)
                                policy_loss = -torch.min(surr1, surr2).mean()

                                value_loss = F.mse_loss(new_values, ret_b)
                                entropy = dist.entropy().mean()

                                loss = policy_loss + 0.5 * value_loss - beta * entropy
                                loss.backward() 
                                company2_agent.optimizer.step()
                                training_steps += 1
                                episodic_losses.append(loss.item())

                                entropies.append(entropy.item())
                                kl_divergences.append((old_lp_b - new_logprobs).mean().item())

                        # Compute avg rewards for company 2
                        avg_rewards_all_agents_company2 = float(sum(rewards)/len(rewards)) * 1000

                        # Reset the memory for the next episode
                        memory.reset()

                # Reset the environment
                END_EPISODE = True

            if not TRAINING and len(env.agents) < 1 and all(dones.values()):
                
                for company in companylist:
                    if company == 'company1':
                        memory = memory_company1
                        beta = beta_company1
                        # agent data
                        agent_data = memory.agent_data
                        all_rewards = []

                        # Company #1 Data Preparation
                        for agent in agent_data.keys():
                            # Extract data for the current agent
                            agent_rewards = torch.tensor(agent_data[agent]['rewards'][:-1], dtype=torch.float32).reshape(-1, 1) 
                            all_rewards.append(agent_rewards)              

                        rewards = torch.cat(all_rewards, dim=0)

                        # Compute avg rewards for company 1
                        avg_rewards_all_agents_company1 = float(sum(rewards)/len(rewards)) * 1000

                        # Reset the memory for the next episode
                        memory.reset()

                    elif company == 'company2':
                        memory = memory_company2
                        beta = beta_company2
                        # agent data
                        agent_data = memory.agent_data
                        all_rewards = []

                        # Company #2 Data Preparation
                        for agent in agent_data.keys():
                            # Extract data for the current agent
                            agent_rewards = torch.tensor(agent_data[agent]['rewards'][:-1], dtype=torch.float32).reshape(-1, 1)
                            all_rewards.append(agent_rewards)

                        rewards = torch.cat(all_rewards, dim=0)

                        # Compute avg rewards for company 2
                        avg_rewards_all_agents_company2 = float(sum(rewards)/len(rewards)) * 1000

                        # Reset the memory for the next episode
                        memory.reset()
                # Compute the average rewards for each company
            if not TRAINING and len(env.agents) < 1 and all(dones.values()):
                END_EPISODE = True
        
        # calculate sum of agent_counters
        total_agent_steps = sum(env.agent_counter.values())
        if episode == 0:
            MAX_STEPS_SO_FAR = total_agent_steps
        else:
            MAX_STEPS_SO_FAR = max(MAX_STEPS_SO_FAR, total_agent_steps)
        # print(f"Total agent steps: {total_agent_steps}")

        # calculate max agent counter
        max_agent_counter = max(env.agent_counter.values())
        # print(f"Max agent time: {float(max_agent_counter*env.dt/60):.2f} min")

        N_success = env.get_num_successful_agents()
        # print(f"Number of successful agents: {N_success}")

        # Update the learning rate
        if TRAINING:
            company1_agent.update_learning_rate(episode)
            company2_agent.update_learning_rate(episode)

        # Calculate the time taken for the episode
        end_time = time.time()
        elapsed_time = end_time - start_time     

        # print out the agent counters from env.agent_counter = {agent: count}
        # print(f"Agent counters: {env.agent_counter}")
        # if len(dones) > (MAX_NUM_AGENTS - 1):

        total_steps = sum(env.agent_counter.values())
        if total_steps == 0:
            total_steps = 1
            print("Sth went wrong in the environment. Total steps = 0. Setting it to 1.")

        # collided agents
        # collided_agents_info = env.collided_agents

        # # finding the sum subrewards of each agent during a given episode
        # episode_rewards = env.each_agent_rewards
        # sum_episode_rewards = env.sum_episode_rewards

        # SumRewards = {}
        # for agent, rewards in episode_rewards.items():
        #     sum_distance_rewards = 0
        #     sum_velocity_rewards = 0
        #     sum_action_rewards = 0
        #     sum_time_rewards = 0
        #     for reward in rewards:
        #         sum_distance_rewards += reward[0]
        #         sum_velocity_rewards += reward[1]
        #         sum_action_rewards += reward[2]
        #         sum_time_rewards += reward[3]
        #     SumRewards[agent] = [sum_distance_rewards, sum_velocity_rewards, sum_action_rewards, sum_time_rewards]

        # sum_rewards_all_agents = 0
        # sum_distance_rewards_all_agents = 0
        # sum_velocity_rewards_all_agents = 0
        # sum_action_rewards_all_agents = 0
        # sum_time_rewards_all_agents = 0
        # for agent in SumRewards.keys():
        #     sum_distance_rewards_all_agents += SumRewards[agent][0]
        #     sum_velocity_rewards_all_agents += SumRewards[agent][1]
        #     sum_action_rewards_all_agents += SumRewards[agent][2]
        #     sum_time_rewards_all_agents += SumRewards[agent][3]

        # avg_distance_rewards_all_agents = sum_distance_rewards_all_agents / total_steps * counter
        # avg_velocity_rewards_all_agents = sum_velocity_rewards_all_agents / total_steps * counter
        # avg_action_rewards_all_agents = sum_action_rewards_all_agents / total_steps * counter
        # avg_time_rewards_all_agents = sum_time_rewards_all_agents / total_steps * counter
        # sum_rewards_all_agents = sum_episode_rewards #sum_distance_rewards_all_agents + sum_velocity_rewards_all_agents #+ sum_action_rewards_all_agents + sum_time_rewards_all_agents

        # avg_rewards_all_agents = sum_episode_rewards/total_steps * counter
        # # avg_rewards_all_agents = sum_episode_rewards/counter * 100.0

        # Save the best model after 100 episodes
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if TRAINING and num_of_collisions < 1:
            if avg_rewards_all_agents_company1 >= best_avg_rewards_company1:
                best_avg_rewards_company1 = avg_rewards_all_agents_company1
                print(f"New best average rewards: {best_avg_rewards_company1}. Saving the model...")

                company1_agent.save(save_path=f'{script_dir}/trained_models/{MODEL_NAME_COMPANY1}')  # Save the best model
            
            if avg_rewards_all_agents_company2 >= best_avg_rewards_company2:
                best_avg_rewards_company2 = avg_rewards_all_agents_company2
                print(f"New best average rewards: {best_avg_rewards_company2}. Saving the model...")
                company2_agent.save(save_path=f'{script_dir}/trained_models/{MODEL_NAME_COMPANY2}')

        # avg_episode_actions = {agent: float(np.mean(actions)) for agent, actions in episode_actions.items()}

        # # Average of avg_episode_actions
        # avg_action = np.mean(list(avg_episode_actions.values()))

        # # Average loss per episode
        # avg_loss = np.mean(episodic_losses)

        # # Average entropy per episode
        # avg_entropy = np.mean(entropies)
        # avg_kl_divergence = np.mean(kl_divergences)

        # # Format `sum_rewards` to 4 decimal places
        # avg_rewards_formatted = f"{avg_rewards_all_agents:.4f}"

        # # Format `SumRewards` dictionary values to 4 decimal places
        # # SumRewards_formatted = {agent: [f"{reward:.4f}" for reward in rewards] for agent, rewards in SumRewards.items()}
        # sum_subrewards_formatted = [f"{sum_distance_rewards_all_agents:.4f}", f"{sum_action_rewards_all_agents:.4f}", f"{sum_time_rewards_all_agents:.4f}"]
        # avg_subrewards_formatted = [f"{avg_distance_rewards_all_agents:.4f}", f"{avg_action_rewards_all_agents:.4f}", f"{avg_time_rewards_all_agents:.4f}"]
        # # Format `avg_episode_actions` dictionary values to 4 decimal places
        # avg_episode_actions_formatted = {agent: f"{avg_action:.4f}" for agent, avg_action in avg_episode_actions.items()}

        # # Foramt avg_loss to 4 decimal places
        # avg_loss = f"{avg_loss:.4f}"

        # Save Scenario
        # if episode == 1:
        #     env.record_scenario()

        # Print the formatted results
        # print(f"\nEpisode {episode}>> Collisions: {num_of_collisions}, Avg Rewards: {avg_rewards_formatted}, Avg Subrewards: {avg_subrewards_formatted}, Sum Subrewards: {sum_subrewards_formatted}, Avg Loss: {avg_loss}, Elapsed Time: {elapsed_time:.2f}, LR: {ppo_agent.new_lr:.5f}, Beta: {beta:.4f}, Training Steps: {training_steps}.")
        print(f"\nEpisode {episode}>> N_success: {N_success}, "
                f"Collisions: {num_of_collisions}, "
                f"Avg Rewards Company 1: {avg_rewards_all_agents_company1:.4f}, "
                f"Avg Rewards Company 2: {avg_rewards_all_agents_company2:.4f}, "
                f"Elapsed Time: {elapsed_time:.2f}, "
                f"Max agent time: {float(max_agent_counter*env.dt/60):.2f} (min).")
        # print(f"Average Actions: {avg_episode_actions_formatted}.")

        # print(f"Episode Counter:{counter}, Collided Agents: {collided_agents_info}, Average Actions: {avg_episode_actions_formatted}.")

        if WANDB:
            wandb.log({
                "Average Rewards Company 1": avg_rewards_all_agents_company1,
                "Average Rewards Company 2": avg_rewards_all_agents_company2,
                # "Sum Distance Rewards": sum_distance_rewards_all_agents,
                # "Sum Action Rewards": sum_action_rewards_all_agents,
                # "Sum Time Rewards": sum_time_rewards_all_agents,
                "N_success": N_success,
                "Collisions": num_of_collisions,
                # "Average Actions": avg_action,
                # "entropy": avg_entropy,
                # "KL Divergence": avg_kl_divergence,
                # "Average Loss": float(avg_loss),
            })

        # else:
        #     print(f"Episode {episode}: No agents left in the environment. Ending episode.")
        #     break


    # Finish wandb run
    if WANDB:
        wandb.finish()



