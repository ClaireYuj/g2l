import math
import random
import shutil
from datetime import datetime
import gc
from thop import profile

from tqdm import tqdm

from algorithm.DDPGFollower import DDPGFollower
from algorithm.FollowerPolicy import FollowerPolicy
from algorithm.IPPOLeader import IPPO
from env.MP_HRL_Env import Env as Env
from algorithm.LeaderPolicy import LeaderPolicy
from algorithm.GCNLeader import GCNLeader
from algorithm.LSTMFollower import LSTMFollower
from algorithm.TransFollower import TransFollower
from algorithm.RNNFollower import RNNFollower
from algorithm.MambaFollower import MambaFollower
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

import psutil

# 获取当前进程

from baselines.CCMADDPG import CCMADDPG

#####################  hyper parameters  ####################
CHECK_EPISODE = 5
SAVE_PER_EPISODE = 50
 #10
MAX_EP_STEPS = 100 #3000
TEXT_RENDER = False
SCREEN_RENDER = False
CHANGE = True
SLEEP_TIME = 0.1




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


def trainer(dataset_index= "KAIST",  model_name = 'STberg', algo_name = "td3", leader_graph = 'e2e', memory_capacity = 800, n_candidate_edge=3,LEARNING_MAX_EPISODE = 50, use_model=0, user_num=25, leader_name=''):
    # dataset_loc = {"KAIST": {"DATASET": "KAIST", "LOCATION": "KAIST_MID", "ES_LENGTH": 600, "ES_WIDTH": 600},
    #                "TaxiSZ": {"DATASET": "TaxiSZ", "LOCATION": "TaxiSZ", "ES_LENGTH": 200, "ES_WIDTH": 200}}
    dataset_loc = {"KAIST": {"DATASET": "KAIST", "LOCATION": "KAIST_MID", "ES_LENGTH": 600, "ES_WIDTH": 600},
                   "TaxiSZ": {"DATASET": "TaxiSZ", "LOCATION": "TaxiSZ", "ES_LENGTH": 200, "ES_WIDTH": 200}}
    # dataset_loc = [("KAIST", "KAIST_MID"), ("TaxiSZ", "TaxiSZ")]

    DATASET = dataset_loc[dataset_index]["DATASET"]  # "KAIST", "TaxiSZ"
    LOCATION = dataset_loc[dataset_index]["LOCATION"]  # "KAIST_MID", "TaxiSZ"
    ES_LENGTH = dataset_loc[dataset_index]["ES_LENGTH"]
    ES_WIDTH = dataset_loc[dataset_index]["ES_WIDTH"]
    n_candidate_edge = n_candidate_edge
    env = Env(dataset=DATASET, location=LOCATION, es_length=ES_LENGTH, es_width=ES_WIDTH, n_candidate_edge=n_candidate_edge, USER_NUM=user_num)
    device = torch.device("cuda:0")
    # make directory
    dir = './output/' + leader_name+model_name +  "/" + DATASET + "/" +"G2L_" + str(n_candidate_edge) + "/"
    dir_name = dir + str(env.user_num) + 'u' + str(int(env.edge_num)) + 'e' + str(
        memory_capacity) + 'mem' + str(LEARNING_MAX_EPISODE) + "epoch"
    print("writing to dir:", dir_name)
    model_dir = dir_name+"/model/"
    if not (os.path.isdir(dir_name)):
        os.makedirs(model_dir)
    # model_name = 'MAberg'  # STberg, Transberg(steer), MAberg, MALSTMberg, MAMBAberg, RNNberg,

    # s_dim:r_dim+e_dim(edge weightys)+2(user's loc)
    s_dim, r_dim, b_dim, o_dim, e_dim, u_dim, mobi_dim, r_bound, b_bound, task_inf, limit, location, mob_var, pred_len, n_candidate_edge = env.get_inf()
    if leader_name == 'IPPO':
        leader_model = IPPO(s_dim=s_dim, r_dim=r_dim, b_dim=b_dim, o_dim=o_dim, e_dim=e_dim, u_dim=u_dim,
                                mobi_dim=mobi_dim, n_candidate_edge=n_candidate_edge,
                                r_bound=r_bound, b_bound=b_bound,batch_size=32,
                                memory_capacity=memory_capacity)
    if leader_name == 'GCN':
        leader_model = GCNLeader(s_dim=s_dim, r_dim=r_dim, b_dim=b_dim, o_dim=o_dim, e_dim=e_dim, u_dim=u_dim,
                                    mobi_dim=mobi_dim, n_candidate_edge=n_candidate_edge,
                                    r_bound=r_bound, b_bound=b_bound, batch_size=8,
                                    memory_capacity=memory_capacity, graph_mode=leader_graph)
    else:
        leader_model = LeaderPolicy(s_dim=s_dim, r_dim=r_dim, b_dim=b_dim, o_dim=o_dim, e_dim=e_dim, u_dim=u_dim,
                                mobi_dim=mobi_dim, n_candidate_edge=n_candidate_edge,
                                r_bound=r_bound, b_bound=b_bound,batch_size=8,
                                memory_capacity=memory_capacity, graph_mode=leader_graph)

    # follow state_dim: task_size(1), ue_loc(2)
    # follow obs_dim: candidate_edge_id(1), candidate_res(1),candidate_loc(2)
    follow_state_dim = 3 +  n_candidate_edge
    follow_obs_dim = 3 * (n_candidate_edge+1) # candidate_state and self_offloading state
    # follow_state_dim = 3 + e_dim
    # follow_obs_dim = 3 * (e_dim + 1)  # candidate_state and self_offloading state
    # if model_name == "steer":
    #     followers_model = FollowerPolicy(state_dim=follow_state_dim, e_dim=e_dim,action_dim=1, n_agent=u_dim, seq_len=mobi_dim,
    #                                      obs_dim=follow_obs_dim, memory_capacity=memory_capacity,
    #                                      candidate_choice=n_candidate_edge)

    if model_name == 'Transberg':  # Trasnformer
        # followers_model = FollowerPolicy(state_dim=follow_state_dim, action_dim=1, n_agent=u_dim, seq_len=mobi_dim,
        #                                  obs_dim=follow_obs_dim, memory_capacity=memory_capacity,
        #                                  candidate_choice=e_dim)
        followers_model = TransFollower(state_dim=follow_state_dim, action_dim=1, n_agents=u_dim, seq_len=mobi_dim,e_dim=n_candidate_edge,
                                        obs_dim=follow_obs_dim, memory_capacity=memory_capacity,
                                        candidate_choice=n_candidate_edge - 1, action_lower_bound=-1,
                                        action_higher_bound=n_candidate_edge - 1, )
    elif model_name == 'MAberg':

        followers_model = DDPGFollower(n_agents=u_dim, state_dim=follow_state_dim + follow_obs_dim, action_dim=1,
                                       e_dim=n_candidate_edge,
                                       seq_len=mobi_dim, action_lower_bound=-1,
                                       action_higher_bound=n_candidate_edge - 1, memory_capacity=memory_capacity)
    elif model_name == "MALSTMberg":
        followers_model = LSTMFollower(n_agents=u_dim, state_dim=follow_state_dim + follow_obs_dim, action_dim=1,e_dim=n_candidate_edge,
                                       seq_len=mobi_dim, action_lower_bound=-1,
                                       action_higher_bound=e_dim - 1, memory_capacity=memory_capacity)
    elif model_name == "RNNberg":
        followers_model = RNNFollower(n_agents=u_dim, state_dim=follow_state_dim + follow_obs_dim, action_dim=1,e_dim=n_candidate_edge,
                                      seq_len=mobi_dim, action_lower_bound=-1,
                                      action_higher_bound=n_candidate_edge - 1, memory_capacity=memory_capacity)
    elif model_name == "MAMBAberg":
        # pass
        followers_model = MambaFollower(n_agents=u_dim, state_dim=follow_state_dim + follow_obs_dim, action_dim=1,e_dim=n_candidate_edge,
                                        seq_len=mobi_dim, action_lower_bound=-1,
                                        action_higher_bound=n_candidate_edge - 1, batch_size=4, memory_capacity=memory_capacity)
    print("follower model:", followers_model)
    print(
        f's_dim:{s_dim}, r_dim:{r_dim}, b_dim:{b_dim}, o_dim:{o_dim}, e_dim:{e_dim},u_dim:{u_dim},r_bound:{r_bound}, b_bound:{b_bound}, task_inf:{task_inf}, limit:{limit}, location:{location}  ')
    print(f"USER NUM:{env.user_num}, edge num:{env.edge_num}")



    if use_model > 0:
        leader_model.load_model(dir_name+"/best_model/best_leader_model_"+str(use_model)+'.pth')
        followers_model.load_model(dir_name+"/best_model/best_followers_model_"+str(use_model)+'.pth')
        step_rewards = open(dir_name + '/step_reward_'+str(use_model)+'.txt', 'a')
        f = open(dir_name + '/record_'+str(use_model)+'.txt', 'a')
    else:
        if (os.path.isdir(dir_name)):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)
        step_rewards = open(dir_name + '/step_reward_' + str(use_model) + '.txt', 'w')
        f = open(dir_name + '/record_' + str(use_model) + '.txt', 'w')

    # ddpg = DDPG(s_dim, r_dim, b_dim, o_dim,e_dim, u_dim, mobi_dim, r_bound, b_bound)

    r_var = 1  # control exploration
    o_var = 1  # o_var
    follow_action_var = 2
    total_reward = []
    leader_reward = []
    follower_reward = []
    best_reward = 0
    # penalty = np.zeros((LEARNING_MAX_EPISODE+1, 5))
    penalty = []
    r_v, b_v = [], []
    var_reward = []
    done = False
    CHANGE = False
    max_rewards = 0
    episode = 0
    var_counter = 0
    process = psutil.Process(os.getpid())


    epoch_inf = []
    # while var_counter < LEARNING_MAX_EPISODE:
    # write the record

    f.write('time(s):' + str(MAX_EP_STEPS) + '\n\n')
    f.write('user_number:' + str(r_dim) + '\n\n')
    f.write('edge_number:' + str(int(o_dim / r_dim)) + '\n\n')
    f.write('limit:' + str(limit) + '\n\n')
    f.write('task information:' + '\n')
    f.write(task_inf + '\n\n')
    rewards_all_in_step = []
    f_rewards_all_in_step = []
    l_rewards_all_in_step = []
    mem_epoch = max(int(memory_capacity//MAX_EP_STEPS), 1)
    for mem in range(mem_epoch):
        s = env.reset()
        fin_req_rate_list = []
        rewards = []
        f_rewards = []
        eff_req = []
        while len(penalty) <= mem_epoch:
            penalty.append([0] * 7)
        for i in tqdm(range(MAX_EP_STEPS), desc=f"Replay buffer {mem}:", ncols=100, ascii=True):
            # Start MDP
            # Leader
            with torch.no_grad():
                a, leader_val = leader_model.choose_actions(
                    s.to(device))  # a = [agent, seq, R+P] [u_dim, seq, e_dim+e_dim*u_dim]
            a = a.cpu()
            # choose the candidates of followers

            shape_a = leader_model.action_shaping(a)  # shape_a = [agent, seq, candidate_edge_num]
            # a = exploration(a, o_dim, o_var, e_dim-1)  # the num of edge to choose offload is e_dim, but it in the list [0...e_dim-1] (len=e_dim)

            # Follow
            follows_state, obs_s = env.communicate2follows(shape_a)
            follows_s = torch.cat([follows_state, obs_s],
                                  dim=-1)  # (agent, seq, task_size(1)+user loc(2)+action(candidate_num)+candidate_resource+(cn*1)+candidate_loc(c_n*2))

            output_action, follower_val = followers_model.choose_actions(
                follows_s.to(device))  # get the candidate or the local?

            follow_action = env.trans_follower_action_to_edge(shape_a, output_action)
            # store the transition parameter
            s_, r, foll_r, fin_req, req_count, fin_req_rate, fin_req_in_time, latency_penalty, edge_overload, energy_penalty = env.ddpg_step_forward(
                follow_action, r_dim, b_dim)
            if not torch.isnan(fin_req_rate):
                a_, a_log_ = leader_model.choose_actions(s_.to(device), algo_name=algo_name)  # a = [R P]
                shaped_a_ = leader_model.action_shaping(a_)
                follows_s_, obs_s_ = env.communicate2follows(shaped_a_)
                follows_s_ = torch.cat([follows_s_, obs_s_], dim=-1)
                if (i+1) % MAX_EP_STEPS == 0:
                    done = True
                else:
                    done = False
                leader_model.memory.store_transition(s, a, r / 1000, s_, leader_val, done)
                followers_model.memory.store_transition(follows_s, output_action, foll_r / 100, follows_s_,
                                                        follower_val, done)
                fin_req_rate_list.append(fin_req_rate)
                rewards.append(r.sum().item())
                f_rewards.append(foll_r.sum().item())
                eff_req = fin_req_in_time.sum().item()
            s = s_


            penalty[mem][0] = fin_req
            penalty[mem][1] = req_count
            penalty[mem][3] = latency_penalty
            penalty[mem][4] = energy_penalty
            penalty[mem][5] = edge_overload
            penalty[mem][6] += eff_req
            # in the end of the episode
            ## use mean
        reward_leader = np.sum(rewards)
        flops, time_cost, param = followers_model.get_flops()

        penalty[mem][2] = np.mean(fin_req_rate_list)
        reward_np = np.sum(f_rewards)
        # follower_reward[episode] = penalty[episode][0] # 0: fin_req, 2:fin_req_rate
        r_v.append(r_var)
        b_v.append(o_var)
        print("===================================================================================================")
        total_r = reward_leader + reward_np
        total_reward.append(total_r)
        print('ReplayBuffer:%3d' % mem, ' Reward: %5d ' % total_r,
              ' Leader reward: %5d' % reward_leader,
              ' Follower reward: %5d' % reward_np)
        print('###  r_var: %.2f ' % r_var, 'o_var: %.2f ' % o_var, ' ### finish request: ', penalty[mem][0],
              ' total request: ', penalty[mem][1], ' request finished rate: ', penalty[mem][2],
              'finish req in time: ', penalty[episode][6], 'time penalty: ', penalty[mem][3],
              'edge overload: ', penalty[mem][5], ' energy penalty: ', penalty[mem][4])
        string_ep = 'ReplayBuffer:%3d' % mem + ' Reward: %5d' % total_r + ' Leader Reward: %5d' % \
                    reward_leader + ' Follower reward: %5d' % reward_np + '\n'
        string_detail = '###  r_var: %.2f ' % r_var + 'o_var: %.2f ' % o_var + ' ### finish request: ' + str(
            penalty[mem][0]) + 'finish req in time: ' + str(
            penalty[mem][6]) + ' total request: ' + str(
            penalty[mem][1]) + ' request finished rate: ' + str(
            penalty[mem][2]) + 'time penalty: ' + str(penalty[mem][3]) + 'edge overload: ' + str(
            penalty[mem][5]) + ' energy penalty: ' + str(penalty[mem][4])
        f.write(string_ep + string_detail + '\n')
    print("\nstart learning\n")
    penalty = []

    re_load_model = False
    while episode < LEARNING_MAX_EPISODE:
        # if re_load_model:
        #     re_load_model = False
        #     leader_model.load_model(model_dir + "/leader_model" + '.pth')
        #     followers_model.load_model(model_dir + "/followers_model" + '.pth')
        #     leader_model.save_model(model_dir + 'leader_model' + ".pth")
        #     followers_model.save_model(model_dir + 'followers_model' + ".pth")
        training_episode = episode + use_model + 1
        while len(penalty) <= episode:
            penalty.append([0] * 7)
        start_time = datetime.now()  # start time for one episode
        s = env.reset()

        leader_loss = []
        follower_loss = []
        fin_req_rate_list = []
        rewards = []
        f_rewards = []

        # initialize
        # s = env.reset()
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

            # time.sleep(SLEEP_TIME)
            # render
            if SCREEN_RENDER:
                env.screen_demo()
            if TEXT_RENDER and j % 30 == 0:
                env.text_render()

            # Start MDP
            # Leader
            a, leader_val = leader_model.choose_actions(
                s.to(device))  # a = [agent, seq, R+P] [u_dim, seq, e_dim+e_dim*u_dim]
            a = a.cpu()
            # choose the candidates of followers

            shape_a = leader_model.action_shaping(a) #  shape_a = [agent, seq, candidate_edge_num]
            # a = exploration(a, o_dim, o_var, e_dim-1)  # the num of edge to choose offload is e_dim, but it in the list [0...e_dim-1] (len=e_dim)

            # Follow
            follows_state, obs_s = env.communicate2follows(shape_a)
            follows_s = torch.cat([follows_state, obs_s],
                                  dim=-1)  # (agent, seq, task_size(1)+user loc(2)+action(candidate_num)+candidate_resource+(cn*1)+candidate_loc(c_n*2))

            output_action, follower_val = followers_model.choose_actions(
                follows_s.to(device))  # get the candidate or the local?

            follow_action = env.trans_follower_action_to_edge(shape_a, output_action)
            # store the transition parameter
            s_, r, foll_r, fin_req, req_count, fin_req_rate, fin_req_in_time, latency_penalty, edge_overload, energy_penalty = env.ddpg_step_forward(
                follow_action, r_dim, b_dim)
            latency_reward = torch.tensor(math.tan(math.pi / latency_penalty))
            if not torch.isnan(fin_req_rate):
                a_, a_log_ = leader_model.choose_actions(s_.to(device), algo_name=algo_name)
                shaped_a_ = leader_model.action_shaping(a_)
                follows_s_, obs_s_ = env.communicate2follows(shaped_a_)
                follows_s_ = torch.cat([follows_s_, obs_s_], dim=-1)
                if j == MAX_EP_STEPS - 1:
                    done = True
                else:
                    done = False
                leader_model.memory.store_transition(s, a, r / 1000, s_, leader_val, done)
                followers_model.memory.store_transition(follows_s, output_action, foll_r/100, follows_s_,
                                                        follower_val, done)

                fin_req_rate_list.append(fin_req_rate)
                rewards.append(r.sum().item())
                f_rewards.append(foll_r.sum().item())
                eff_req = fin_req_in_time.sum().item()
                rewards_all_in_step.append(r.sum().item()+foll_r.sum().item())


            # else:
            #     print("fin req is null")

            # learn
            # if leader_model.pointer == leader_model.memory_capacity:
            # if leader_model.memory.mem_cntr % leader_model.memory.mem_size == 0:
            #
            #     if leader_model.memory.mem_cntr == leader_model.memory.mem_size:
            # if leader_model.memory.mem_cntr > leader_model.memory.mem_size:
            if algo_name == 'ppo':
                l_loss = leader_model.learn_ppo()
                f_loss = followers_model.train()
            else:
                l_loss = leader_model.learn_td3()
                f_loss = followers_model.train()
            # leader_loss.append(l_loss.item())
            # follower_loss.append((f_loss.item()))
            if CHANGE:
                factor = random.uniform(0.9997, 0.9999)
                r_var *= factor  # .99999
                o_var *= factor  # .99999
                mob_var *= factor
            penalty[episode][0] = fin_req
            penalty[episode][1] = req_count
            penalty[episode][3] = latency_penalty
            penalty[episode][4] = energy_penalty
            penalty[episode][5] = edge_overload
            penalty[episode][6] += eff_req
            step_rewards.write(str(r.sum().item() + foll_r.sum().item()) + "," + str(r.sum().item()) + "," + str(
                foll_r.sum().item()) + '\n')
            # replace the state
            s = s_

            # sum up the reward
            # leader_reward[episode] += r.cpu().numpy()

            # in the end of the episode
            if j == MAX_EP_STEPS - 1:
                done = True
                ## use mean

                reward_np = np.sum(rewards)
                # leader_reward[episode] = rewards[-1]
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

                flops, time_cost, param = followers_model.get_flops()

                print(
                    "===================================================================================================")
                total_r = leader_reward[episode] + follower_reward[episode]
                total_reward.append(total_r)
                print('Episode:%3d' % training_episode, ' Reward: %5d ' % total_r,
                      ' Leader reward: %f' % leader_reward[episode],
                      ' Follower reward: %f' % follower_reward[episode])
                print('###  r_var: %.2f ' % r_var, 'o_var: %.2f ' % o_var, ' ### finish request: ', penalty[episode][0],
                      ' total request: ', penalty[episode][1], ' request finished rate: ', penalty[episode][2],
                      'finish req in time: ', penalty[episode][6], 'time penalty: ', penalty[episode][3],
                      'edge overload: ', penalty[episode][5], ' energy penalty: ', penalty[episode][4],
                      ' ### using time: ', diff_time, 'flops: ', flops, "param:", param, 'one action cost time:',time_cost)
                # print(f"Leader Loss:{np.mean(leader_loss)}, Follower loss:{np.mean(follower_loss)}")
                string_ep = 'Episode:%3d' % training_episode + ' Reward: %5d' % total_r + ' Leader Reward: %5d' % leader_reward[
                    episode] + ' Follower reward: %5d' % follower_reward[episode] + '\n'
                string_detail = '###  r_var: %.2f ' % r_var + 'o_var: %.2f ' % o_var + ' ### finish request: ' + str(
                    penalty[episode][0]) + 'finish req in time: ' + str(
                    penalty[episode][6]) + ' total request: ' + str(
                    penalty[episode][1]) + ' request finished rate: ' + str(
                    penalty[episode][2]) + 'time penalty: ' + str(penalty[episode][3]) + 'edge overload: ' + str(
                    penalty[episode][5]) + ' energy penalty: ' + str(penalty[episode][4]) + ' ### using time: ' + str(
                    diff_time)+'flops: '+str(flops)+"param:"+str(param)+ 'one action cost time:'+str(time_cost)
                f.write(string_ep + string_detail + '\n')
                # epoch_inf.append(string_ep+string_detail)
                # var_reward = var_reward.detach().cpu().numpy()  # 转换为 numpy 数组
                # variation change
                print("var reward type:", type(var_reward), "var reward:", var_reward)
                # if total_r > best_reward:
                #     best_reward = total_r
                #     print("=========BEST MODEL "+str(training_episode)+"===================")
                #     if not os.path.exists(dir_name+"/best_model/"):
                #         os.makedirs(dir_name+"/best_model/")
                #     leader_model.save_model(dir_name + "/best_model/" + 'best_leader_model_' + str(training_episode) + ".pth")
                #     followers_model.save_model(
                #         dir_name + "/best_model/" + 'best_followers_model_' + str(training_episode) + ".pth")
                # if np.mean(follower_loss) > 90:
                #     CHANGE = True
                if training_episode % SAVE_PER_EPISODE == 0:
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    leader_model.save_model(model_dir+'leader_model'+".pth")
                    followers_model.save_model(model_dir+'followers_model'+".pth")
                    re_load_model = True


                if var_counter >= CHECK_EPISODE and np.mean(var_reward[-CHECK_EPISODE:]) >= max_rewards:
                    CHANGE = True
                    var_counter = 0
                    max_rewards = np.mean(var_reward[-CHECK_EPISODE:])
                    var_reward = []
                else:
                    CHANGE = False
                    var_counter += 1
                torch.cuda.empty_cache()


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

    # plot the reward
    fig_reward = plt.figure()
    plt.plot([i + 1 for i in range(len(rewards_all_in_step))], rewards_all_in_step)
    plt.xlabel("episode")
    plt.ylabel("rewards")
    fig_reward.savefig(dir_name + '/rewards_step.png')
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
    f.write("the mean of the total rewards in the last" + str(CHECK_EPISODE) + " epochs:" + str(
        np.mean(total_reward[-CHECK_EPISODE:])) + "\n")
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
    f.write("the mean of the leader rewards in the last" + str(CHECK_EPISODE) + " epochs:" + str(
        np.mean(leader_reward[-CHECK_EPISODE:])) + "\n")

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
    f.write("the mean of the followers rewards in the last" + str(CHECK_EPISODE) + " epochs:" + str(
        np.mean(follower_reward[-CHECK_EPISODE:])))
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
    del leader_model
    del followers_model
    gc.collect()  # 调用垃圾回收器
    step_rewards.close()
    f.close()


if __name__ == "__main__":
        is_KAIST = True
        isMamba = True
        random.seed(42)
        np.random.seed(42)
        user_num_list = [20,40,50]

        if isMamba:
            for num in user_num_list:
                if num == 20:
                    trainer(dataset_index="TaxiSZ", model_name='MAMBAberg', algo_name="ppo", leader_graph='e2e',
                            memory_capacity=800, LEARNING_MAX_EPISODE=50, use_model=0, user_num=num, n_candidate_edge=7)
                else:
                    trainer(dataset_index="KAIST", model_name='MAMBAberg', algo_name="ppo", leader_graph='e2e',
                            memory_capacity=800, LEARNING_MAX_EPISODE=50, use_model=0,user_num=num,n_candidate_edge=7)
                    trainer(dataset_index="TaxiSZ", model_name='MAMBAberg', algo_name="ppo",  leader_graph='e2e',
                    memory_capacity = 800, LEARNING_MAX_EPISODE = 50, use_model = 0, user_num=num,n_candidate_edge=7)
        else:

            if is_KAIST:
                for num in user_num_list:
                    # if num == 30:
                    #     trainer(dataset_index="KAIST", model_name='MAberg', algo_name="ppo", leader_graph='e2e', memory_capacity=800,
                    #     LEARNING_MAX_EPISODE=50, use_model=0,user_num=num, leader_name='IPPO')
                    trainer(dataset_index="KAIST", model_name='steer', algo_name="ppo", leader_graph='e2e', memory_capacity=800,
                        LEARNING_MAX_EPISODE=50, use_model=0,user_num=num,n_candidate_edge=3)


            else:
                for num in user_num_list:
                    if num ==  40:
                        pass
                    # if num == 30:
                    #     trainer(dataset_index="TaxiSZ", model_name='MAberg', algo_name="ppo", leader_graph='e2e',
                    #         memory_capacity=800, LEARNING_MAX_EPISODE=50, use_model=0, user_num=num, leader_name='IPPO')
                    trainer(dataset_index="TaxiSZ", model_name='MAberg', algo_name="ppo", leader_graph='e2e',
                        memory_capacity=800, LEARNING_MAX_EPISODE=50, use_model=0, user_num=num)

