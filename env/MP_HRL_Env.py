import random
import numpy as np
import math
from src.render import Demo as Demo
import torch
import env.UE_Agent as UE_Agent

#####################  hyper parameters  ####################

in_step = 10 # each UE pred per 10 steps
out_step = 5
USE_TORCH = True
USE_STATE_ENCODE = True
USER_NUM = 25 #15
EDGE_NUM = 4
EDGE_SERVER_LENGTH = 800 # x
EDGE_SERVER_HEIGHT = 800 # y
# EDGE_SERVER_LENGTH = 400 # x
# EDGE_SERVER_HEIGHT = 400 # y

CANDIDATE_EDGE_NUM = 3
LIMIT = 4
user_req_pool_size = 200
edge_req_pool_size = 2000
MAX_EP_STEPS = 3000
TXT_NUM = 70
r_ratio = 0.8 # 0.063
r_bound = 1000 # UE_Agent:EDGE_CAPACITY
task_uplimit = 400
b_bound = 1000 # 带宽用于估算传输速率

# r_bound = 1e9 * 0.063
# b_bound = 1e9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn


def BandwidthTable(edge_num):
    BandwidthTable = np.zeros((edge_num, edge_num))
    for i in range(0, edge_num):
        for j in range(i+1, edge_num):
                # BandwidthTable[i][j] = 1e9
                BandwidthTable[i][j] = b_bound
    # print(f"Bandwidth table:{BandwidthTable}")
    return BandwidthTable



# def OffloadingTable(user_num, edge_num):
#     OffloadingTable = np.zeros((user_num, edge_num))
#     for i in range(0, user_num):
#         for j in range(i + 1, edge_num):
#             # BandwidthTable[i][j] = 1e9
#             OffloadingTable[i][j] = 0
#     return OffloadingTable

def two_to_one(two_table):
    """
    two dimmension transfer to 1 dimmension
    :param two_table:
    :return:
    """
    one_table = two_table.flatten() # [[1,2][3,4]] -> [1,2,3,4]
    return one_table


def generate_leader_state(U, edge_seq_R, E, x_min, model ,weights, seq_len, use_edge_seq_R=False):
    """
     S 是生成的系统状态 (state) 向量.[ n_agent,seq_len(PRED),(Resource(len(E)), priority(len(E)), User's loc)]
    用于表示当前系统中用户设备 (UE)、边缘服务器 (MEC)、以及连接资源的整体状态。
    :param two_table:
    :param U:
    :param E:
    :param x_min:
    :param y_min:
    :return:
    """
    # initial
    # s = torch.zeros((len(U), seq_len, len(E)+len(E)+2))
    s = torch.zeros((len(U), seq_len, len(E)+len(E)))

    # transform
    count = 0
    # available resource of each edge server
    if use_edge_seq_R:
        for r_id in range(edge_seq_R.shape[1]):
            s[:, :, count] = edge_seq_R[:, r_id]/(r_bound*2)
            count += 1
    else:
        for edge in E:
            # a = s[:, :, count]
            s[:, :, count] = edge.capability/(r_bound*2) # 用 r_bound*10 归一化resource。
            count += 1



    dists = model.generate_dist(U, E, weights)
    s[:, :, count:count+len(E)] = torch.from_numpy(dists) / len(E)

    # weights = model.generate_weights(U, E, weights)
    # s[:, :, count:count+len(E)] = torch.from_numpy(weights) / len(E)

    count+=len(E)


    return s.cpu()



def generate_follower_state(leader_action, U, edge_seq_R, R, E, x_min, use_edge_seq_R=False):
    # s = torch.zeros((len(U), seq_len, CANDIDATE_EDGE_NUM+2))
    n_agent, seq_len, n_cand = leader_action.shape
    cand_info = torch.zeros((n_agent, seq_len, n_cand*3))
    ue_info = torch.zeros((n_agent, seq_len, 3))
    cand_res = torch.zeros_like(leader_action) # candidate edge resource
    cand_loc = torch.zeros((n_agent, seq_len, n_cand*2)) # candidate edge loc (id*2)
    u_loc = torch.zeros((len(U), seq_len, 2))

    task_size = torch.zeros((n_agent, seq_len, 1))
    for i_agent in range(n_agent):
       for i_seq in range(seq_len):
           for i_edge in range(n_cand):
               edge_id = int(leader_action[i_agent, i_seq, i_edge].item())
               ## [cand_1_res, cand_1_loc, cand_2_res, cand_2_loc]
               if use_edge_seq_R:
                   cand_info[i_agent, i_seq, i_edge * 3] = edge_seq_R[i_seq, edge_id] / r_bound
               else:
                   cand_info[i_agent, i_seq, i_edge * 3] = R[edge_id] / r_bound

               cand_info[i_agent, i_seq, i_edge*3+1] = (E[edge_id].loc[0] + abs(x_min)) / 1e5
               cand_info[i_agent, i_seq, i_edge*3+2] = ( E[edge_id].loc[1]  + abs(x_min)) / 1e5
           u_id = i_agent
           task = torch.full((seq_len, 1), U[u_id].req_pool[0].tasktype.process_loading / task_uplimit)
           mob = torch.from_numpy( U[u_id].pre_locs)
           u_loc[u_id, :, -2] = (mob[:, 0] + abs(x_min)) / 1e5
           u_loc[u_id, :, -1] = (mob[:, 1] + abs(x_min)) / 1e5
           task_size[u_id, :, :] = task
           ue_info[i_agent, i_seq, 0] = U[u_id].resource/100
           ue_info[i_agent, i_seq, 1] = (U[u_id].pre_locs[i_seq][0] + abs(x_min)) / 1e5
           ue_info[i_agent, i_seq, 2] = (U[u_id].pre_locs[i_seq][0] + abs(x_min)) / 1e5

               # # [cand_1_res, cand_2_res], [cand_1_loc, cand_2_loc]
               # cand_res[i_agent, i_seq, i_edge] = R[edge_id] / r_bound
               # cand_loc[i_agent, i_seq, i_edge*2] = (E[edge_id].loc[0] + abs(x_min)) / 1e5
               # cand_loc[i_agent, i_seq, i_edge*2+1] = ( E[edge_id].loc[1]  + abs(x_min)) / 1e5

    # u_id = 0
    # for user in U:
    #     u_id = user.user_id
    #     task = torch.full((seq_len, 1), user.req_pool[0].tasktype.process_loading/task_uplimit)
    #     mob = torch.from_numpy(user.pre_locs)
    #     u_loc[u_id, :, -2] = (mob[:, 0] + abs(x_min)) / 1e5
    #     u_loc[u_id, :, -1] = (mob[:, 1] + abs(x_min)) / 1e5
    #     task_size[u_id, :, :] = task
    #     u_id+=1
    # s = torch.cat((task_size, u_loc), dim=-1)  # (n_agent, seq, [task, loc]

    s = torch.cat((task_size, u_loc, leader_action,), dim=-1)  # (n_agent, seq, [task, loc]
    # obs = torch.cat((cand_res, cand_loc), dim=-1)  # (n_agent, seq, [leader_action, cand_res, cand_loc]
    obs = torch.cat((cand_info, ue_info), dim=-1)  # (n_agent, seq, [leader_action, cand_res, cand_loc]

    return s, obs
    # return leader_action, u_loc
        # count += 2



"""
def generate_action(R, B, O):

    # resource
    a = np.zeros(USER_NUM + USER_NUM + EDGE_NUM * USER_NUM)
    a[:USER_NUM] = R / r_bound
    # bandwidth
    a[USER_NUM:USER_NUM + USER_NUM] = B / b_bound
    # offload
    base = USER_NUM + USER_NUM
    for user_id in range(USER_NUM):
        a[base + int(O[user_id])] = 1
        base += EDGE_NUM
    return a
"""

def get_minimum_maximum(dataset, location):
    max_y_over = []
    min_y_over = []
    max_x_over = []
    min_x_over = []

    cal = np.zeros((1, 2))
    for data_num in range(TXT_NUM):
        data_name = str("%03d" % (data_num + 1))  # plus zero
        file_name = dataset + "_30sec_" + data_name + ".txt"
        file_path = "./data/" + location + "/" + file_name
        f = open(file_path, "r")
        f1 = f.readlines()
        # get line_num
        line_num = 0
        for line in f1:
            line_num += 1
        # collect the data from the .txt
        data = np.zeros((line_num, 2))
        index = 0
        for line in f1:
            data[index][0] = line.split()[1]  # x
            data[index][1] = line.split()[2]  # y
            index += 1
        # put data into the cal
        cal = np.vstack((cal, data))
        # print("data NUM:", data_num+1)

        if min(data[:, 0]) < -2000:
            min_x_over.append(data_num+1)
        if min(data[:, 1]) < -2000:
            min_y_over.append(data_num+1)
        if max(data[:, 0]) > 2000:
            max_x_over.append(data_num + 1)
        if max(data[:, 1]) > 2000:
            max_y_over.append(data_num + 1)


    return min(cal[:, 0]), min(cal[:, 1]), max(cal[:, 0]), max(cal[:, 1])


def set_fixed_edge_loc(edge_length, edge_height, x_min, y_min, x_max, y_max):
    global EDGE_NUM
    # The range(length, height) of each edge server is fixed
    edge_count_x = math.ceil((x_max - x_min)/edge_length)
    edge_count_y = math.ceil((y_max - y_min)/edge_height)
    EDGE_NUM = edge_count_x * edge_count_y
    e_l = np.zeros((EDGE_NUM, 2))
    for y in range(edge_count_y):
        for x in range(edge_count_x):
            edge_id = x + y * edge_count_x
            e_l[edge_id][0] = x_min + x * edge_length + edge_length / 2
            e_l[edge_id][1] = y_min + y * edge_height + edge_height / 2
    # print("edge num:", EDGE_NUM, "edge_id:", edge_id)

    return EDGE_NUM, e_l

# def proper_edge_loc(edge_num):
#
#     # initial the e_l
#     e_l = np.zeros((edge_num, 2)) # e_l dim: [edge_num, 2], edge_loaction
#     # calculate the mean of the data
#     group_num = math.floor(TXT_NUM / edge_num)
#     edge_id = 0
#     for base in range(0, group_num*edge_num, group_num):
#         for data_num in range(base, base + group_num):
#             data_name = str("%03d" % (data_num + 1))  # plus zero
#             file_name = DATASET + "_30sec_" + data_name + ".txt"
#             file_path = "./data/" + LOCATION + "/" + file_name
#             # f = open(file_path, "r")
#             # f1 = f.readlines()
#             with open(file_path, "r") as f:
#                 f1 = f.readlines()
#             line_num = len(f1)
#             # get line_num and initial data
#             # line_num = 0
#             # for line in f1:
#             #     line_num += 1
#             data = np.zeros((line_num, 2))
#             # collect the data from the .txt
#             index = 0
#             for line in f1:
#                 data[index][0] = line.split()[1]  # x
#                 data[index][1] = line.split()[2]  # y
#                 index += 1
#             # stack the collected data
#             if data_num % group_num == 0:
#                 cal = data
#             else:
#                 cal = np.vstack((cal, data))
#         e_l[edge_id] = np.mean(cal, axis=0)
#         edge_id += 1
#
#     return e_l


#############################Policy#######################


class priority_policy():
    # def generate_priority(self, U, E, priority): # rank
    #     for user in U:
    #         # get a list of the offloading priority
    #         dist = np.zeros(EDGE_NUM)
    #         for edge in E:
    #             dist[edge.edge_id] = np.sqrt(np.sum(np.square(user.loc[0] - edge.loc)))
    #         dist_sort = np.sort(dist)
    #         for index in range(EDGE_NUM):
    #             priority[user.user_id][index] = np.argwhere(dist == dist_sort[index])[0]
    #     return priority

    # def generate_priority(self, U, E, priority):
    #     """
    #     generate the edge priority for each user
    #     rank by the avg of next 10 steps
    #     :param U:
    #     :param E:
    #     :param priority: 2d <user_num-edge_num>
    #     :return:
    #     """
    #     next_dists = np.zeros((USER_NUM,EDGE_NUM))
    #     for user in U:
    #         # get a list of the offloading priority
    #         dist = np.zeros(EDGE_NUM)
    #         for prob_loc_index in range(len(user.next_locs)):
    #             for edge in E:
    #                 if prob_loc_index == 0:
    #                     dist[edge.edge_id] = 0
    #                 # print(f"user.next_locs[prob_loc_index]:{user.next_locs[prob_loc_index]}, edge.loc:{edge.loc}")
    #                 # dist[edge.edge_id] += np.sqrt(np.sum(np.square(user.next_locs[prob_loc_index] - edge.loc))) * 0.1
    #                 dist[edge.edge_id] += np.sqrt(np.sum(np.square(user.next_locs[prob_loc_index] - edge.loc))) * 0.1
    #
    #         dist_sort = np.sort(dist)
    #         for index in range(EDGE_NUM):
    #             priority[user.user_id][index] = np.argwhere(dist == dist_sort[index])[0]
    #     return priority

    def generate_dist(self, U, E, dists):
        """

        Generate the edge weight for each user as a 3D tensor.
        weight = edge_num -  priority
        Rank by the avg of next 10 steps.
        :param U: List of users
        :param E: List of edge servers
        :param priority: 3D <user_num, edge_num, next_loc_priority>
        :return: Updated priority tensor
        """
        # Initialize priority tensor with appropriate shape
        # priority = np.zeros((len(U), 20, len(E)))  # (user_num, edge_num, next_loc_priority)

        for user in U:
            # Get a list of the offloading priorities
            for prob_loc_index in range(len(user.pre_locs)):
                dist = np.zeros(len(E))
                for edge in E:
                    # Calculate distance from user location to edge location
                    dist[edge.edge_id] += np.sqrt(np.sum(np.square(user.pre_locs[prob_loc_index] - edge.loc))+1e-8)
                #
                # # Sort edges based on distance to rank them
                # dist_sort_indices = np.argsort(dist)
                # dis_weight_indices = np.argsort(dist_sort_indices)
                # dis_weight_indices = len(E) - dis_weight_indices

                # Update the priority tensor for the current user and prob_loc_index
                dists[user.user_id, prob_loc_index, :] = dist

        return dists
    # def generate_weights(self, U, E, weights):
    #     """
    #
    #     Generate the edge weight for each user as a 3D tensor.
    #     weight = edge_num -  priority
    #     Rank by the avg of next 10 steps.
    #     :param U: List of users
    #     :param E: List of edge servers
    #     :param priority: 3D <user_num, edge_num, next_loc_priority>
    #     :return: Updated priority tensor
    #     """
    #     # Initialize priority tensor with appropriate shape
    #     # priority = np.zeros((len(U), 20, len(E)))  # (user_num, edge_num, next_loc_priority)
    #
    #     for user in U:
    #         # Get a list of the offloading priorities
    #         for prob_loc_index in range(len(user.pre_locs)):
    #             dist = np.zeros(len(E))
    #             for edge in E:
    #                 # Calculate distance from user location to edge location
    #                 dist[edge.edge_id] += np.sqrt(np.sum(np.square(user.pre_locs[prob_loc_index] - edge.loc)))
    #
    #             # Sort edges based on distance to rank them
    #             dist_sort_indices = np.argsort(dist)
    #             dis_weight_indices = np.argsort(dist_sort_indices)
    #             dis_weight_indices = len(E) - dis_weight_indices
    #
    #             # Update the priority tensor for the current user and prob_loc_index
    #             weights[user.user_id, prob_loc_index, :] = dis_weight_indices
    #
    #     return weights

    def indicate_edge(self, O, U, weights):
        """
        get self.offloading
        :param O:
        :param U:
        :param priority:
        :return:
        """
        edge_limit = np.ones((EDGE_NUM)) * LIMIT
        for user in U:
            for index in range(EDGE_NUM):
                O[user.user_id] = random.randint(0, EDGE_NUM-1)
                # if edge_limit[int(weights[user.user_id][0][index])-1] - 1 >= 0:
                #     edge_limit[int(weights[user.user_id][0][index])-1] -= 1
                #     O[user.user_id] = weights[user.user_id][0][index] -1
                #     break
        return O
    # def indicate_edge(self, O, U, priority):
    #     """
    #     get self.offloading
    #     :param O:
    #     :param U:
    #     :param priority:
    #     :return:
    #     """
    #     edge_limit = np.ones((EDGE_NUM)) * LIMIT
    #     for user in U:
    #         for index in range(EDGE_NUM):
    #             if edge_limit[int(priority[user.user_id][index])] - 1 >= 0:
    #                 edge_limit[int(priority[user.user_id][index])] -= 1
    #                 O[user.user_id] = priority[user.user_id][index]
    #                 break
    #     return O



    def bandwidth_update(self, O, table, B, U, E):
        for user in U:
            share_number = 1
            ini_edge = int(user.req.edge_id)
            target_edge = int(O[user.req.user_id])
            # no need to migrate
            if ini_edge == target_edge:
                B[user.req.user_id] = 0
            # provide bandwidth to migrate
            else:
                # share bandwidth with user from migration edge
                for user_id in E[target_edge].user_group:
                    if O[user_id] == ini_edge:
                        share_number += 1
                # share bandwidth with the user from the original edge to migration edge
                for ini_user_id in E[ini_edge].user_group:
                    if ini_user_id != user.req.user_id and O[ini_user_id] == target_edge:
                        share_number += 1
                # allocate the bandwidth
                B[user.req.user_id] = table[min(ini_edge, target_edge)][max(ini_edge, target_edge)] / (share_number+2)

        return B

#############################Env###########################

class Env():
    def __init__(self, dataset="KAIST", location="KAIST_MID", es_length=1500, es_width=1500, n_candidate_edge=3, USER_NUM=USER_NUM):
        self.location = location
        self.dataset = dataset
        self.use_edge_seq_R = True
        self.step = 10
        self.seq_len = in_step
        # self.pred_len = out_step
        self.pred_len = in_step

        self.n_candidate_edge = n_candidate_edge

        self.user_latency_penalty_factor = 0.1
        self.migration_latency_penalty_factor = 0.1
        self.energe_penalty_factor = 0.01
        self.time = 0
        self.x_min, self.y_min, self.x_max, self.y_max = get_minimum_maximum(dataset=self.dataset, location=self.location)
        # self.edge_num = EDGE_NUM  # the number of servers:10
        self.edge_num, self.e_l = set_fixed_edge_loc(es_length, es_width,
                                                self.x_min, self.y_min, self.x_max, self.y_max)
        self.user_num = USER_NUM  # the number of users:60
        # define environment object
        self.reward_all = []
        self.U = [] # user
        self.fin_req_count = 0
        self.fin_req_in_time = 0
        self.prev_count = 0
        self.total_req = 0
        self.rewards = 0
        self.R = np.zeros((self.user_num)) # resource:self.user_num
        # self.O = np.zeros((self.edge_num)) #offloading:self.user_num
        self.O = np.zeros((self.user_num)) #offloading:self.user_num
        self.B = np.zeros((self.edge_num)) #bandwidth:self.user_num
        self.table = BandwidthTable(self.edge_num)
        self.weights = np.zeros((self.user_num, self.seq_len, self.edge_num))
        # self.state_encoder = StateEncoder()
        self.E = []
        self.edge_step_R = torch.zeros((self.pred_len, self.edge_num))
        data_num = random.sample(list(range(TXT_NUM)), self.user_num) # choose the ue from txt(ue<92)
        for i in range(self.user_num):
            new_user = UE_Agent.UE(i, data_num[i], self.n_candidate_edge, self.dataset, self.location, seq_len=self.seq_len)
            self.U.append(new_user)
        # self.edge_step_R = torch.zeros((PRED_STEP, self.edge_num))


        # self.edge_num, e_l = set_fixed_edge_loc(EDGE_SERVER_LENGTH, EDGE_SERVER_HEIGHT,
        #                                         self.x_min, self.y_min, self.x_max, self.y_max)
        # e_l = proper_edge_loc(self.edge_num) # choose proper edge server?
        for i in range(self.edge_num):
            new_e = UE_Agent.EdgeServer(i, self.e_l[i, :],r_bound=r_bound) # create edge server on proper location
            self.E.append(new_e)
        self.model = 0
        self.text_render_file = open("./output/ini_records.txt", "w")

    def get_inf(self):
        # s_dim
        self.reset()
        s = generate_leader_state(self.U, self.edge_step_R,self.E, self.x_min, self.model, self.weights, self.seq_len, use_edge_seq_R=self.use_edge_seq_R)

        # s_dim = s.size
        if USE_STATE_ENCODE:
            s_dim = s.size(2) # tensor
        else:
            s_dim = s.size # npArray

        r_dim = len(self.E)
        b_dim = len(self.E)
        e_dim = len(self.E)
        u_dim = len(self.U)
        mobi_dim = self.seq_len
        o_dim = self.edge_num * len(self.U)

        # maximum resource
        r_bound = self.E[0].capability

        # maximum bandwidth
        b_bound = float(1000)
        # b_bound = b_bound.astype(np.float32)

        # task size
        task = UE_Agent.TaskType()
        task_inf = task.task_inf()

        # mob var
        mob_range = self.y_max - self.y_min
        edge_range = int(mob_range /round(math.sqrt(self.edge_num)))
        mob_var = mob_range / edge_range

        return s_dim, r_dim, b_dim, o_dim, e_dim, u_dim, mobi_dim, r_bound, b_bound, task_inf, LIMIT, self.location, mob_var, self.pred_len, self.n_candidate_edge

    def reset(self):

        # reset time
        start_time = random.randint(300, 1050)
        self.time = start_time # start from 5
        self.total_req = 0
        # reward
        self.reward_all = []
        # user
        # self.U = []
        self.fin_req_count = 0
        self.prev_count = 0



        # Resource
        self.R = np.zeros((self.edge_num))
        # Offloading
        # self.O = np.zeros((self.user_num)) # edge server corresponding to user

        self.O = np.zeros((self.user_num)) # edge server corresponding to user
        # bandwidth
        self.B = np.zeros((self.edge_num))
        for k in range(self.user_num):
            self.U[k].release()
        # self.R = np.full((self.edge_num), r_bound)
        # bandwidth table
        self.table = BandwidthTable(self.edge_num)
        # server

        for i in range(self.edge_num):
            new_e = self.E[i]
            new_e.release()
            self.R[i] = new_e.capability
            self.edge_step_R[:, i] = new_e.capability
            # self.text_render_file.write("edge " + str(new_e.edge_id) + "'s loc:"+ str(new_e.loc)+"'resource:"+ str(new_e.capability)+"\n")

            # print("edge", new_e.edge_id, "'s loc:\n", new_e.loc)

        # model
        self.model = priority_policy()

        # initialize the request
        self.weights = self.model.generate_dist(self.U, self.E, self.weights)
        self.O = self.model.indicate_edge(self.O, self.U, self.weights)

        for user in self.U:
            self.generate_req_in_pool(user.user_id, self.O[user.user_id])
            # user.generate_request(self.O[user.user_id])

        # return generate_state(self.table, self.U, self.E, self.x_min, self.state_encoder, self.model, self.priority)
        return generate_leader_state(self.U, self.edge_step_R, self.E, self.x_min, self.model, self.weights, self.seq_len, use_edge_seq_R=self.use_edge_seq_R)

    def trans_follower_action_to_edge(self, candidate_edges, follow_action):
        n_agent, seq_len, n_cand = follow_action.shape
        edge_action = torch.zeros((n_agent, seq_len, n_cand))
        for i_agent in range(n_agent):
            for i_seq in range(seq_len):
                for i_edge in range(n_cand):
                    try:
                        candidate_choice = int(follow_action[i_agent, i_seq, i_edge].item())
                        edge_id = int(candidate_edges[i_agent, i_seq, candidate_choice].item())
                        edge_action[i_agent, i_seq, i_edge] = edge_id
                    except Exception as e:
                        print("candidate_choice:", candidate_choice)
                        print(e)
        return edge_action

    def generate_req_in_pool(self, user_id, edge_id):
        ## update the req in both user's req_pool, and edge_req_pool
        edge_id = int(edge_id)
        user_id = int(user_id)
        # print("cur req num:", self.get_fin_req_num(self.U), "cur total:", self.get_total_req_num(self.U))
        if len(self.U[user_id].req_pool) < user_req_pool_size:
        # if len(self.U[user_id].req_pool) < user_req_pool_size and len(self.E[edge_id].req_pool) < edge_req_pool_size:
            req = self.U[user_id].generate_request(edge_id)
            self.E[int(edge_id)].req_pool.append(req)
            if user_id not in self.E[edge_id].user_group:
                self.E[edge_id].user_group.append(user_id)
        else:
            self.U[user_id].req_count+=1


    def ddpg_step_forward(self, follow_action, r_dim, b_dim, is_deterministic=False):
        # release the bandwidth
        n_agent, seq, _ = follow_action.shape
        total_edge_capacity_cost = 0
        fin_migration_timer = 0
        fin_req_in_time = 0

        # self.table = BandwidthTable(self.edge_num)
        follower_action_choice = follow_action.detach()
        # self.fin_req_count = 0

        # release the resource
        # for edge in self.E:
        #     edge.release()

        # self.B = np.full((self.edge_num), b_bound)
        rewards = torch.zeros((n_agent, seq, 1))
        # f_rewards = torch.zeros((seq, 1))
        fin_req_rewards = torch.zeros((n_agent, seq, 1))
        follower_reward = torch.zeros((n_agent, seq, 1))
        eff_req_rewards = torch.zeros((n_agent, seq, 1))
        energy_tensor = torch.zeros((n_agent, seq, 1))
        latency_tensor = torch.zeros((n_agent, seq, 1))
        total_req_num = 0
        latency_penalty = 0
        energy_penalty = 0

        # offloading update
        # base = r_dim + b_dim
        for step in range(seq):
            for user_id in range(self.user_num):
                # self.U[user_id].candidate_edge = a[user_id, :self.n_candidate_edge, 0]
                # self.U[user_id].candidate_edge = follower_action_choice[user_id, :self.n_candidate_edge, 0]

                self.U[user_id].candidate_edge = follower_action_choice[user_id, step, :]


                # self.generate_req_in_pool(user_id, self.O[user_id])
                ## initial the req that state=0
                for req in self.U[user_id].req_pool:

                    if req.state == 0:
                        req.timer = -1

                        for edge in self.U[user_id].candidate_edge:
                            action = int(edge.item())
                            req.timer += 1
                            if action < 0:
                                self.U[user_id].self_process(req)
                            else:

                                self.O[user_id] = action
                                req.edge_id = self.O[user_id]
                                offload_success = self.request_offloading(req, action)

                                req = self.U[user_id].request_update(req, step)
                                # if not offload_success:
                                #     print(f"user_id:{user_id} fail to  edge:{action}")
                                if offload_success:
                                    # self.U[user_id].req.state = 0
                                    self.U[user_id].request_update(req, step)
                                    break
                                # if not offload_success:
                                # # print(f"user_id:{user_id} fail to offload")
                                #     self.generate_req_in_pool(user_id, self.O[user_id])
            edge_overload = 0

            for edge in self.E:
                self.R = edge.process_request(self.R)
            for edge in self.E:
                self.R = edge.process_request(self.R)
                edge_overload += edge.overload

            req_timer = 0
            # request update
            for user in self.U:
                # update the state of the request
                for req in user.req_pool:
                    # self.req = req
                    req = user.request_update(req, step)
                    req_timer += req.timer
                    total_edge_capacity_cost += req.energy
                    energy_tensor[user.user_id,step,0] += req.energy
                    ## correct offload
                    if req.state == 4:
                        # rewards
                        req.state = 5  # request turn to "disconnect"
                        user.finish_request(req)
                        self.fin_req_count += 1 # successful offloading times + 1
                        fin_req_rewards[user.user_id, step, 0] = rewards[user.user_id, step, 0] + 1

                        if req.timer < 1:
                            follower_reward[user.user_id, step, 0] = follower_reward[user.user_id, step, 0] + (1 - req.timer)*0.5
                            if req.timer < 0.5:
                                fin_req_in_time += 1
                                eff_req_rewards[user.user_id, step, 0] = eff_req_rewards[user.user_id, step, 0] + 1
                                f_r = (1 - req.timer)*2 if (1 - req.timer) < 0.5 else (1 - req.timer)*3
                                follower_reward[user.user_id, step, 0] = follower_reward[user.user_id, step, 0] + f_r
                        elif req_timer > 3:
                            latency_tensor[user.user_id, step, 0] = latency_tensor[user.user_id, step, 0] + 0.5
                            if req_timer > 5:
                                latency_tensor[user.user_id, step, 0] = latency_tensor[user.user_id, step, 0] + 1

                        # self.R[int(user.req.edge_id)] += user.req.tasktype.task_size
                        # self.generate_req_in_pool(user.user_id, self.O[user_id])
                    # else:
                    #     print("req fail:",req)


                    # elif req.timer >= CANDIDATE_EDGE_NUM or req.state == 5:
                    #     # print("pop the request because req.timer is ", req.timer)
                    #     user.switch_request()


                        # user.generate_request(self.O[user.user_id])
                    # it has already finished the request


            # for edge in self.E:
            #     edge.maintain_request(self.R, self.U)
            #     self.table, mig_time = edge.migration_update(self.O, self.B, self.table, self.U, self.E)
            #     total_edge_capacity_cost += (1 - edge.capability/EDGE_CAPACITY)
            #     fin_migration_timer += mig_time
            # self.R = self.model.resource_update(self.R, self.E, self.U)
            # self.B = self.model.bandwidth_update(self.O, self.table, self.B, self.U, self.E)
            total_edge_capacity_cost /= len(self.E)
            total_req_num = self.get_total_req_num(self.U)
            # rewards
            latency_penalty += (self.user_latency_penalty_factor * req_timer) + (self.migration_latency_penalty_factor * fin_migration_timer)
            energy_penalty += (total_edge_capacity_cost * self.energe_penalty_factor + edge_overload*self.energe_penalty_factor)
            ## calculate the finished req
            cur_fin = self.get_fin_req_num(self.U)
            # fin_req = cur_fin - prev_fin
            fin_req = self.get_fin_req_num(self.U)
            # if fin_req - latency_penalty < 0:
            #     r = 0
            # else:
            #     # r = (fin_req-latency_penalty) ** 2
            #     # r = fin_req - latency_penalty
            #     r = math.tan(math.pi/latency_penalty) + energy_penalty
            #     # r = math.log(((1+latency_penalty)/(1-latency_penalty))**2)
            # r = fin_req - edge_overload * self.energe_penalty_factor + fin_req_in_time
            # r = fin_req_in_time - energy_penalty
            # r = fin_req_rewards + follower_reward - energy_penalty
            r = (fin_req_rewards + eff_req_rewards - latency_tensor - energy_tensor*self.energe_penalty_factor - edge_overload*self.energe_penalty_factor/(self.user_num*seq))

            # print("fin_req_in_time:", fin_req_in_time, " energy_penalty:",energy_penalty, "r: ",r)

            # self.rewards = torch.tensor(0) if r <= 0 else torch.tensor(r) # relu
            fin_req = torch.tensor(fin_req)
            fin_req_in_time = torch.tensor(fin_req_in_time)
            total_req_num = torch.tensor(total_req_num)


            for agent in range(rewards.size(0)):
                # rewards[agent, step,:] = torch.tensor(0) if r[agent, step, :] + self.U[agent].fin_req *0.005 <= 0 else torch.tensor(r[agent, step, :])+ self.U[agent].fin_req *0.005
                rewards[agent, step,:] = torch.tensor((r[agent, step, :])/100) if r[agent, step, :]<= 0 else torch.tensor(r[agent, step, :])

            # self.prev_count = self.fin_req_count
            # print(f"fin_req:{fin_req}, latency_penalty:{latency_penalty}")
            # print(f" self.total_req:{self.total_req}, total_req_num:{total_req_num}, cur_total_req:{cur_total_req}, prev:{self.prev_count}")

            # every user start to move
            if self.time % self.step == 0:
                for user in self.U:
                    user.mobility_update(self.time)
            # print("cur req num:", self.get_fin_req_num(self.U), "cur total:", self.get_total_req_num(self.U))

            # print("fin req rate:", fin_req_rate)

            # update time
            self.time += 1
            if self.use_edge_seq_R:
                self.edge_step_R[step ,:] = torch.from_numpy(self.R)
            for user_id in range(len(self.U)):
                self.generate_req_in_pool(user_id, self.O[user_id])

            # return s_, r
        total_req_num = self.get_total_req_num(self.U)
        fin_req_rate = (self.get_fin_req_num(self.U) / total_req_num) * 100
        fin_req_rate = torch.tensor(fin_req_rate)

        # follower_reward = follower_reward

        return (generate_leader_state(self.U, self.edge_step_R,self.E, self.x_min, self.model, self.weights, self.seq_len, use_edge_seq_R=self.use_edge_seq_R),
                rewards,  follower_reward, fin_req, total_req_num, fin_req_rate, eff_req_rewards, latency_penalty, edge_overload, energy_penalty)


    def communicate2follows(self, shaping_a):
        return generate_follower_state(shaping_a, self.U, self.edge_step_R, self.R, self.E, self.x_min, use_edge_seq_R=self.use_edge_seq_R)

    def get_total_req_num(self, U):
        sum = 0
        for u in U:
            sum += u.req_count
        return sum

    def get_fin_req_num(self, U):
        sum = 0
        for u in U:
            sum += u.fin_req
        return sum

    def update_offload_edge(self, user_id, edge_id):
        user_id = int(user_id)
        edge_id = int(edge_id)
        # pre_edge_id = int(self.U[user_id].req.edge_id)
        # self.E[pre_edge_id].user_group.remove(user_id)
        # self.U[user_id].req.edge_id = edge_id
        self.O[user_id] = edge_id
        # self.E[edge_id].user_group.append(user_id)

    def request_offloading(self, req, edge_id):

        if self.R[edge_id] > req.tasktype.req_u2e_size: # offloading success
            # self.R[edge_id] -= req.tasktype.req_u2e_size
            return True
        else:
            return False

        # if self.R[edge_id] > self.U[user_id].req.tasktype.task_size: # offloading success
        #     self.R[edge_id] -= self.U[user_id].req.tasktype.task_size
        #     self.O[user_id] = edge_id
        #     return True
        # else:
        #     return False




    def text_render(self):
        f = self.text_render_file
        # print("f:",f)
        self.text_render_file.write("R:" + str(self.R)+"\n")
        self.text_render_file.write("B:" + str(self.B)+"\n")
        """
        base = USER_NUM +USER_NUM
        for user in range(len(self.U)):
            print("user", user, " offload probabilty:", a[base:base + self.edge_num])
            base += self.edge_num
        """
        self.text_render_file.write("O:" + str(self.O)+"\n")
        for user in self.U:
            self.text_render_file.write("user"+str(user.user_id)+"'s loc:"+str(user.loc)+"\n")
            self.text_render_file.write("request state:"+str(user.req.state)+"\n")
            self.text_render_file.write("edge serve:"+str(user.req.edge_id)+"\n")

        for edge in self.E:
            self.text_render_file.write("edge"+str(edge.edge_id)+"user_group:"+str(edge.user_group)+"\n")
        self.text_render_file.write("reward:" + str(self.rewards))
        self.text_render_file.write(("=====================update=============================="))
        # print("=====================update==============================")
        # f.close()


    def open_text_render_file(self, filename):
        self.text_render_file = open(filename, "a+")

    def close_text_render_file(self):
        self.text_render_file.close()

    def initial_screen_demo(self):
        self.canvas = Demo(self.E, self.U, self.O, MAX_EP_STEPS)

    def screen_demo(self):
        self.canvas.draw(self.E, self.U, self.O)


