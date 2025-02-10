import math
import random
import shutil
from datetime import datetime

from tqdm import tqdm

from algorithm.FollowerPolicy import FollowerPolicy
from baselines.MAGNN import MAGNN
from env.MP_HRL_Env import Env as Env
from baselines.CCMADDPG import CCMADDPG
from baselines.MAPPO import MAPPO
from baselines.IPPO import IPPO
from baselines.IGNN import IGNN
# from baselines.MAMAMBA import MAMAMBA
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch

#####################  hyper parameters  ####################
CHECK_EPISODE = 5
LEARNING_MAX_EPISODE = 1 #10
MAX_EP_STEPS = 100 #3000
TEXT_RENDER = False
SCREEN_RENDER = False
CHANGE = True
SLEEP_TIME = 0.1

random.seed(42)

def exploration(a, o_dim, o_var, o_bound):

    # 资源部分添加噪声
    # a[:r_dim] = torch.clamp(torch.normal(a[:r_dim].float(), r_var), 0, 1) * r_bound
    # a[:r_dim] = a[:r_dim]*(r_bound*10) # return
    # offloading部分添加噪声
    # b = torch.normal(a[:, :, r_dim:r_dim + o_dim].float(), o_var)
    b = torch.normal(a[:, :, :].float(), o_var)

    c = torch.clamp(torch.round(b) , 0, 1) * o_bound
    # a[:, :, r_dim:r_dim + o_dim] = c
    a[:, :, :] = c

    # a[:, :, r_dim:r_dim + o_dim] = torch.clamp(torch.normal(a[:, :, r_dim:r_dim + o_dim].float(), o_var), 0,
    #                                            1) * o_bound

    # a[r_dim+b_dim:] = torch.clamp(torch.normal(a[r_dim+b_dim:], b_var), 0.9, 1) * mob_var
    return a

def trainer(dataset_index= "KAIST",  model_name = 'MADDPG', algo_name = "td3", memory_capacity = 800, LEARNING_MAX_EPISODE = 50, use_model=0, user_num=25):
    dataset_loc = {"KAIST": {"DATASET": "KAIST", "LOCATION": "KAIST_MID", "ES_LENGTH":600, "ES_WIDTH":600},
                   "TaxiSZ": {"DATASET": "TaxiSZ", "LOCATION": "TaxiSZ", "ES_LENGTH":200, "ES_WIDTH":200}}

    DATASET = dataset_loc[dataset_index]["DATASET"] # "KAIST", "TaxiSZ"
    LOCATION = dataset_loc[dataset_index]["LOCATION"] # "KAIST_MID", "TaxiSZ"
    ES_LENGTH = dataset_loc[dataset_index]["ES_LENGTH"]
    ES_WIDTH = dataset_loc[dataset_index]["ES_WIDTH"]
    env = Env(dataset=DATASET, location=LOCATION, es_length=ES_LENGTH, es_width=ES_WIDTH,USER_NUM=user_num)

    # s_dim:r_dim+e_dim(edge weightys)+2(user's loc)
    s_dim, r_dim, b_dim, o_dim, e_dim, u_dim, mobi_dim, r_bound, b_bound, task_inf, limit, location, mob_var, pred_len, n_candidate_edge = env.get_inf()
    follow_state_dim = 3 + e_dim
    follow_obs_dim = 3 * (e_dim + 1)  # candidate_state and self_offloading state
    s_dim = follow_obs_dim+follow_state_dim
    if model_name == 'MADDPG':
        model = CCMADDPG(n_agents=u_dim, e_dim=e_dim,state_dim=s_dim, action_dim=1, seq_len=mobi_dim, action_lower_bound=-1,
                 action_higher_bound=e_dim-1, memory_capacity=memory_capacity)
    elif model_name == 'MAPPO':
        model = MAPPO(n_agents=u_dim, e_dim=e_dim,state_dim=s_dim, action_dim=1, seq_len=mobi_dim, action_lower_bound=-1,
                 action_higher_bound=e_dim-1, memory_capacity=memory_capacity)
    elif model_name == 'IPPO':
        model = IPPO(s_dim=s_dim, r_dim=r_dim, b_dim=b_dim, o_dim=o_dim, e_dim=e_dim, u_dim=u_dim,
                                mobi_dim=mobi_dim, n_candidate_edge=n_candidate_edge,
                                r_bound=r_bound, b_bound=b_bound, batch_size=32,
                                memory_capacity=memory_capacity)
    elif model_name == 'IGNN':
        model = IGNN(s_dim=s_dim, r_dim=r_dim, b_dim=b_dim, o_dim=o_dim, e_dim=e_dim, u_dim=u_dim,
                                mobi_dim=mobi_dim, n_candidate_edge=n_candidate_edge,
                                r_bound=r_bound, b_bound=b_bound,batch_size=2,
                                memory_capacity=memory_capacity)
    elif model_name == 'MAMAMBA':
        model = MAMAMBA(n_agents=u_dim, e_dim=e_dim,state_dim=s_dim, action_dim=1, seq_len=mobi_dim, action_lower_bound=-1,
                 action_higher_bound=e_dim-1, memory_capacity=memory_capacity)
    elif model_name == 'MAGNN':
        model = MAGNN(s_dim=s_dim, r_dim=r_dim, b_dim=b_dim, o_dim=o_dim, e_dim=e_dim, u_dim=u_dim,
                                mobi_dim=mobi_dim, n_candidate_edge=n_candidate_edge,
                                r_bound=r_bound, b_bound=b_bound,batch_size=2,
                                memory_capacity=memory_capacity)

    print(f's_dim:{s_dim}, r_dim:{r_dim}, b_dim:{b_dim}, o_dim:{o_dim}, e_dim:{e_dim},u_dim:{u_dim},r_bound:{r_bound}, b_bound:{b_bound}, task_inf:{task_inf}, limit:{limit}, location:{location}  ')
    print(f"USER NUM:{env.user_num}, edge num:{env.edge_num}")
    device = torch.device("cuda:0")
    # make directory

    dir = './output/' + model_name + "/" + DATASET + "/"
    dir_name = dir + str(env.user_num) + 'u' + str(int(env.edge_num)) + 'e' + str(
        memory_capacity) + 'mem' + str(LEARNING_MAX_EPISODE) + "epoch"
    print("writing to dir:", dir_name)
    if (os.path.isdir(dir_name)):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    # ddpg = DDPG(s_dim, r_dim, b_dim, o_dim,e_dim, u_dim, mobi_dim, r_bound, b_bound)

    r_var = 1  # control exploration
    o_var = e_dim # o_var
    follow_action_var = 2
    leader_reward = []
    total_reward = []
    follower_reward = []
    # penalty = np.zeros((LEARNING_MAX_EPISODE+1, 5))
    penalty = []
    r_v, b_v = [], []
    var_reward = []
    done = False

    max_rewards = 0
    episode = 0
    var_counter = 0
    # write the record
    step_rewards = open(dir_name + '/step_reward_'+str(use_model)+'.txt', 'w')

    f = open(dir_name + '/record'+str(use_model)+'.txt', 'w')
    f.write('time(s):' + str(MAX_EP_STEPS) + '\n\n')
    f.write('user_number:' + str(r_dim) + '\n\n')
    f.write('edge_number:' + str(int(o_dim / r_dim)) + '\n\n')
    f.write('limit:' + str(limit) + '\n\n')
    f.write('task information:' + '\n')
    f.write(task_inf + '\n\n')
    epoch_inf = []
    # while var_counter < LEARNING_MAX_EPISODE:

    while episode < LEARNING_MAX_EPISODE:

        while len(penalty) <= episode:
            penalty.append([0] * 7)
        start_time = datetime.now() # start time for one episode

        leader_loss = []
        # follower_loss = []
        fin_req_rate_list = []
        rewards = []
        f_rewards = []

        # initialize
        x = torch.arange(0, e_dim)
        candi_e = x.view(1, 1, e_dim).expand(u_dim, mobi_dim, e_dim)  # 扩展到 [20, 5, 10]

        s = env.reset()
        follows_state, obs_s = env.communicate2follows(candi_e)
        s = torch.cat([follows_state, obs_s],
                              dim=-1)
        leader_reward.append(0)
        follower_reward.append(0)
        if SCREEN_RENDER:
            env.initial_screen_demo()

        if TEXT_RENDER:
            env.close_text_render_file()
            text_render_file_path = dir_name + "/render_records_episode_" + str(var_counter) + ".txt"
            env.open_text_render_file(text_render_file_path)
        # for j in range(MAX_EP_STEPS): # relate to seq_len
        for j in tqdm(range(MAX_EP_STEPS), desc=f"Episode{episode}", ncols=100, ascii=True):
            time.sleep(SLEEP_TIME)
            # render
            if SCREEN_RENDER:
                env.screen_demo()
            if TEXT_RENDER and j % 30 == 0:
                env.text_render()

            # Start MDP
            # Leader
            # with torch.no_grad():
            a, des_edge, leader_val = model.choose_actions(s.to(device))# a = [agent, seq, R+P] [u_dim, seq, e_dim+e_dim*u_dim]
            a = a.cpu()
            # choose the candidates of followers

            # shape_a = leader_model.action_shaping(a) #  shape_a = [agent, seq, candidate_edge_num]

            # Follow

            # add randomness to action selection for exploration
            # a = exploration(a, o_dim, o_var, e_dim - 1)  # the num of edge to choose offload is e_dim, but it in the list [0...e_dim-1] (len=e_dim)

            # store the transition parameter
            s_, r, foll_r, fin_req, req_count, fin_req_rate, fin_req_in_time, latency_penalty, edge_overload,energy_penalty = env.ddpg_step_forward(des_edge, r_dim, b_dim)
            latency_reward = torch.tensor(math.tan(math.pi/latency_penalty))
            # if leader_model.pointer == leader_model.memory_capacity:
            if model.memory.mem_cntr % model.memory.mem_size == 0:
                done = True
                if  model.memory.mem_cntr == model.memory.mem_size:
                    print("\nstart learning\n")
            if not torch.isnan(fin_req_rate):
                # a_, a_log_ = model.choose_actions(s_.to(device))  # a = [R P]
                follows_s_, obs_s_ = env.communicate2follows(candi_e)
                s_ = torch.cat([follows_s_, obs_s_], dim=-1)

                # model.memory.store_transition(s, a, r/1000+foll_r/100, s_, leader_val, done)
                model.memory.store_transition(s, a, r/1000+foll_r/100, s_, leader_val, done)

                fin_req_rate_list.append(fin_req_rate)
                rewards.append(r.sum().item())
                f_rewards.append(foll_r.sum().item())
                eff_req = fin_req_in_time.sum().item()
            # else:
            #     print("fin req is null")

            # learn

            if model.memory.mem_cntr > model.memory.mem_size:

                l_loss =model.train()

                leader_loss.append(l_loss.item())

            # replace the state
            s = s_
            # sum up the reward
            # leader_reward[episode] += r.cpu().numpy()
            step_rewards.write(str(r.sum().item()+foll_r.sum().item())+","+str(r.sum().item())+","+str(foll_r.sum().item())+'\n')

            penalty[episode][0] = fin_req
            penalty[episode][1] = req_count
            penalty[episode][3] = latency_penalty
            penalty[episode][4] = energy_penalty
            penalty[episode][5] = edge_overload
            penalty[episode][6] += eff_req

            # in the end of the episode
            if j == MAX_EP_STEPS - 1:
                ## use mean
                reward_np = np.sum(rewards)
                leader_reward[episode] = reward_np
                var_reward.append(leader_reward[episode])
                penalty[episode][2] = np.mean(fin_req_rate_list)
                reward_np = np.sum(f_rewards)
                reward_mean = np.mean(f_rewards)
                follower_reward[episode] = reward_np
                # follower_reward[episode] = penalty[episode][0] # 0: fin_req, 2:fin_req_rate
                r_v.append(r_var)
                b_v.append(o_var)
                end_time = datetime.now()  # ed time for one episode
                diff_time = end_time - start_time
                print("===================================================================================================")
                total_r = leader_reward[episode]+follower_reward[episode]
                total_reward.append(total_r)
                print('Episode:%3d' % episode, ' Reward: %5d' % total_r,' Leader reward: %5d'% leader_reward[episode], 'Follower reward: ', follower_reward[episode],)
                print('###  r_var: %.2f ' % r_var,'o_var: %.2f ' % o_var, ' ### finish request: ',  penalty[episode][0], ' total request: ',  penalty[episode][1],' request finished rate: ',  penalty[episode][2], 'finish req in time: ', penalty[episode][6],'time penalty: ',  penalty[episode][3], 'edge overload: ', penalty[episode][5], ' energy penalty: ',  penalty[episode][4], ' ### using time: ',  diff_time)

                string_ep = 'Episode:%3d' % episode + ' Reward: %5d' % total_r +' Leader reward: %5d'% leader_reward[episode] + '\n'
                string_detail = '###  r_var: %.2f ' % r_var + 'o_var: %.2f ' % o_var + ' ### finish request: ' + str(penalty[episode][0])+'finish req in time: '+ str(penalty[episode][6])+ ' total request: '+  str(penalty[episode][1])+' request finished rate: '+  str(penalty[episode][2])+ 'time penalty: '+  str(penalty[episode][3])+ 'edge overload: '+str(penalty[episode][5])+' energy penalty: '+str(penalty[episode][4])+ ' ### using time: '+ str(diff_time)
                f.write(string_ep+string_detail+'\n')
                epoch_inf.append(string_ep+string_detail)
                # var_reward = var_reward.detach().cpu().numpy()  # 转换为 numpy 数组
                # variation change
                print("var reward type:", type(var_reward), "var reward:", var_reward)

                # if np.mean(follower_loss) > 90:
                #     CHANGE = True

                if var_counter >= CHECK_EPISODE and np.mean(var_reward[-CHECK_EPISODE:]) >= max_rewards:
                    CHANGE = True
                    var_counter = 0
                    max_rewards = np.mean(var_reward[-CHECK_EPISODE:])
                    var_reward = []
                else:
                    CHANGE = False
                    var_counter += 1
        # end the episode

        if SCREEN_RENDER:
            env.canvas.tk.destroy()
        episode += 1
    if TEXT_RENDER:
        env.close_text_render_file()

        # plot the reward
    fig_reward = plt.figure()
    plt.plot([i + 1 for i in range(episode)], leader_reward)
    plt.xlabel("episode")
    plt.ylabel("rewards")
    fig_reward.savefig(dir_name + '/leader_rewards.png')
    # plot the follower reward
    fig_reward = plt.figure()
    plt.plot([i + 1 for i in range(episode)], follower_reward)
    plt.xlabel("episode")
    plt.ylabel("rewards")
    fig_reward.savefig(dir_name + '/follower_rewards.png')
    # plot the variance
    # fig_variance = plt.figure()
    # plt.plot([i + 1 for i in range(episode)], r_v, b_v)
    # plt.xlabel("episode")
    # plt.ylabel("variance")
    # fig_variance.savefig(dir_name + '/variance.png')
    #
    # for i in range(episode):
    #     f.write(str(epoch_inf[i]) + '\n')
    print("The figure and file is saved in ", dir_name)
    # mean
    print("the mean of the total rewards in the last", CHECK_EPISODE, " epochs:",
          str(np.mean(total_reward[-CHECK_EPISODE:])))
    f.write("the mean of the total rewards in the last"+str(CHECK_EPISODE)+" epochs:"+str(np.mean(total_reward[-CHECK_EPISODE:]))+"\n")
    f.write("the mean of the total rewards:" + str(np.mean(total_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    # mean
    print("the max value of the total rewards:",
          str(max(total_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the max value of the total rewards:" + str(
        max(total_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    f.write(
        "************************************************************************************************************")
    # mean
    print("the mean of the leader rewards in the last", CHECK_EPISODE, " epochs:",
          str(np.mean(leader_reward[-CHECK_EPISODE:])))
    f.write("the mean of the leader rewards in the last"+str(CHECK_EPISODE)+" epochs:"+str(np.mean(leader_reward[-CHECK_EPISODE:]))+"\n")

    f.write("the mean of the leader rewards:" + str(np.mean(leader_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    # standard deviation
    print("the standard deviation of the leader rewards:", str(np.std(leader_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the standard deviation of the leader rewards:" + str(
        np.std(leader_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    # range
    print("the range of the leader rewards:",
          str(max(leader_reward[-LEARNING_MAX_EPISODE:]) - min(leader_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the range of the leader rewards:" + str(
        max(leader_reward[-LEARNING_MAX_EPISODE:]) - min(leader_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    # max value
    print("the max value of the leader rewards:",
          str(max(leader_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the max value of the leader rewards:" + str(
        max(leader_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    f.write(
        "************************************************************************************************************")
    ### followers reward
    print("the mean of the followers rewards in the last", CHECK_EPISODE, " epochs:",
          str(np.mean(follower_reward[-CHECK_EPISODE:])))
    f.write("the mean of the followers rewards in the last"+ str(CHECK_EPISODE)+" epochs:"+str(np.mean(follower_reward[-CHECK_EPISODE:])))
    f.write("the mean of the followers rewards:" + str(np.mean(follower_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    # standard deviation
    print("the standard deviation of the followers rewards:", str(np.std(follower_reward[-LEARNING_MAX_EPISODE:])))
    f.write(
        "the standard deviation of the followers rewards:" + str(
            np.std(follower_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    # range
    print("the range of the followers rewards:",
          str(max(follower_reward[-LEARNING_MAX_EPISODE:]) - min(follower_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the range of the followers rewards:" + str(
        max(follower_reward[-LEARNING_MAX_EPISODE:]) - min(follower_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    # max value
    print("the max value of the follower rewards:",
          str(max(follower_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the max value of the follower rewards:" + str(
        max(follower_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    step_rewards.close()

    f.close()



if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    isKAIST = True
    user_num_list = [30]
    if isKAIST:
        for num in user_num_list:
            #
            # trainer(dataset_index="KAIST", model_name='MADDPG', algo_name="ppo", memory_capacity=800, LEARNING_MAX_EPISODE=50,
            #         user_num=num)
            trainer(dataset_index="KAIST", model_name='MAPPO', algo_name="ppo", memory_capacity=800, LEARNING_MAX_EPISODE=50, user_num=num)
            # trainer(dataset_index="KAIST", model_name='MAPPO', algo_name="ppo", memory_capacity=800, LEARNING_MAX_EPISODE=50, user_num=num)
            # trainer(dataset_index="KAIST", model_name='IGNN', algo_name="ppo", memory_capacity=800, LEARNING_MAX_EPISODE=50, user_num=num)
            #
            # trainer(dataset_index="KAIST", model_name='IPPO', algo_name="ppo", memory_capacity=800, LEARNING_MAX_EPISODE=50, user_num=num)
    else:
        for num in user_num_list:

            # trainer(dataset_index="TaxiSZ", model_name='MADDPG', algo_name="ppo", memory_capacity=800,
            #      LEARNING_MAX_EPISODE=50,
            #      user_num=num)
            # trainer(dataset_index="KAIST", model_name='IPPO', algo_name="ppo", memory_capacity=20, LEARNING_MAX_EPISODE=50, user_num=num)
            trainer(dataset_index="TaxiSZ", model_name='MAPPO', algo_name="ppo", memory_capacity=800,
                    LEARNING_MAX_EPISODE=50, user_num=num)
            # trainer(dataset_index="TaxiSZ", model_name='MAPPO', algo_name="ppo", memory_capacity=800,
            #      LEARNING_MAX_EPISODE=50, user_num=num)
            # trainer(dataset_index="TaxiSZ", model_name='IGNN', algo_name="ppo", memory_capacity=15, LEARNING_MAX_EPISODE=50, user_num=num)
            #
            # trainer(dataset_index="TaxiSZ", model_name='IPPO', algo_name="ppo", memory_capacity=800, LEARNING_MAX_EPISODE=50, user_num=num)



