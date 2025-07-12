
from itertools import count
from PIL import Image
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TKAgg')
#from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython import display
import random
import numpy as np
import torch
import pandas as pd

def seed_random(num):
    np.random.seed(num)
    random.seed(num)
    torch.manual_seed(num)
    print("random seed: {0}".format(num))

def plot_result(axis, r_data, num_group, performance, save_dir, env_name):
    COLOURS = ['red','blue','green','orange','black','gold','violet','olive','dimgrey','saddlebrown']
    SHADES = ['#f69393', '#5863f5', '#64fa78', '#edc464', '#b8b4ab','#fffd87','#faaaf7','#7fc77b','#b5b5b5','#96634b']
    CON_COLOURS = ['r-','b-','g-','o-','k-','y-','m-','c-','r--','b--']

    #COOPERATIVE REWARDS / individual rewards
    results = []
    results_std = []
    for idx in range(num_group):
        per_agent_reward = [item[idx][0] for item in r_data]
        per_agent_reward_std = [item[idx][1] for item in r_data]
        results.append(per_agent_reward)
        results_std.append(per_agent_reward_std)
    if performance == "R":
        results.append([item[num_group][0] for item in r_data])
        results_std.append([item[num_group][1] for item in r_data])
    
    all_x_axis = axis
    fig, ax = plt.subplots()
    for idxidx in range(num_group):
        label_name = "Group_{0}".format(idxidx+1)
        eval_mean = np.array(results[idxidx])
        eval_std = np.array(results_std[idxidx])
        diff_minus_eval = np.array(eval_mean) - np.array(eval_std)
        diff_plus_eval = np.array(eval_mean) + np.array(eval_std)
        ax.plot(all_x_axis, eval_mean, alpha=0.5, color=COLOURS[idxidx+1], label=label_name, linewidth=1.0) #1.0
        ax.fill_between(all_x_axis, diff_minus_eval.tolist(), diff_plus_eval.tolist(), color=SHADES[idxidx+1], alpha=0.2)

    ax.grid()
    ax.legend(loc='upper left',fontsize=10)
    plt.xlabel('Steps',fontsize=10)
    if performance == "R":
        plt.ylabel('Average Returns',fontsize=10)
    else: #"C"
        plt.ylabel('Number of Collisions',fontsize=10)
    save_name = 'result_' + performance + '_' + env_name +'_fullshare_maddpg.png'
    plt.savefig(save_dir + save_name, dpi=400)
    plt.close()

    if performance == "R":
        fig, ax = plt.subplots()
        label_name = "All"
        eval_mean = np.array(results[-1])
        eval_std = np.array(results_std[-1])
        diff_minus_eval = np.array(eval_mean) - np.array(eval_std)
        diff_plus_eval = np.array(eval_mean) + np.array(eval_std)
        ax.plot(all_x_axis, eval_mean, alpha=0.5, color=COLOURS[0], label=label_name, linewidth=1.0) #1.0
        ax.fill_between(all_x_axis, diff_minus_eval.tolist(), diff_plus_eval.tolist(), color=SHADES[0], alpha=0.2)
        ax.grid()
        ax.legend(loc='upper left',fontsize=10)
        plt.xlabel('Steps',fontsize=10)
        plt.ylabel('Average Returns',fontsize=10)
        save_name = 'result_all' + performance + '_' + env_name + '_fullshare_maddpg.png'
        plt.savefig(save_dir + save_name, dpi=400)
        plt.close()

def save_result(axis, r_data, num_group, name_env, name_alg, name_type, save_dir):
    results = []
    results_std = []
    for idx in range(num_group):
        per_agent_reward = [item[idx][0] for item in r_data]
        per_agent_reward_std = [item[idx][1] for item in r_data]
        results.append(per_agent_reward)
        results_std.append(per_agent_reward_std)
    
    all_x_axis = axis #list(range(len(results[0])))
    S_DATA = dict()
    S_DATA['steps'] = all_x_axis
    for idx in range(num_group):
        name_mean = "group_{0}_mean".format(idx)
        name_std = "group_{0}_std".format(idx)
        S_DATA[name_mean] = results[idx]
        S_DATA[name_std] = results_std[idx]

    df = pd.DataFrame(S_DATA)
    df.to_pickle(save_dir+'data_env_' + name_env + '_alg_' + name_alg + '_type_' + name_type + '.pkl', protocol=4)

def schedule_epsilon(init, min, explore_len, current_step):
    u_epsilon = max( min, current_step*((min-init)/explore_len)+init )
    return u_epsilon
