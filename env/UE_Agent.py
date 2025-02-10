import random

import numpy as np
import torch
import math

#####################  hyper parameters  ####################
# LOCATION = "KAIST_MID"
# DATASET = "KAIST"
USE_TORCH = True


LIMIT = 5
r_ratio = 1 # 0.063
r_bound = 1000
b_bound = 1000 # 带宽用于估算传输速率
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Controller(object):
    def __init__(self, U, E, R):
        self.U = U
        self.R = R
        self.E = E

    def sync_resource(self, e_id, resource):
        self.R[e_id] = resource

    def allocate_resource(self, u_id, resource):
        self.U[u_id].req.resource = resource


class UE(object):
    def __init__(self, user_id, data_num, CANDIDATE_EDGE_NUM, dataset="KAIST", location="KAIST_MID", seq_len=10):
        self.user_id = user_id  # number of the user
        self.dataset = dataset
        self.location = location
        self.loc = np.zeros((1, 2))
        self.num_step = 0  # the number of step
        self.req_count = 0
        self.fin_req = 0
        self.resource_limit = 100

        # calculate num_step and define self.mob
        data_num = str("%03d" % (data_num + 1))  # plus zero
        file_name = self.dataset + "_30sec_" + data_num + ".txt"
        file_path = "./data/" + self.location + "/" + file_name
        f = open(file_path, "r")
        f1 = f.readlines()
        data = 0
        for line in f1:
            data += 1
        self.num_step = data * 10 # calculate the number of records for this UE
        self.mob = np.zeros((self.num_step, 2)) # store the all location of UE
        self.req_pool = []
        # write data to self.mob
        self.resource = self.resource_limit


        now_sec = 0
        for line in f1:
            for sec in range(10):
                self.mob[now_sec + sec][0] = line.split()[1]  # x
                self.mob[now_sec + sec][1] = line.split()[2]  # y
            now_sec += 10
        self.loc[0] = self.mob[0]
        self.pred_mobility_length = seq_len
        # self.next_locs = np.zeros((10, 2)) # 10 step = 5 min
        # self.next_locs = self.get_next_mobility(0) # 10 step = 5 min
        self.pre_locs = self.get_pre_mobility(0)
        self.candidate_edge = np.zeros((CANDIDATE_EDGE_NUM))

    # def get_next_mobility(self, cur_time, sec_in_step=30):
    #     next_step = self.pred_mobility_length
    #     # get next 10 step, while each step is 30 seconds
    #     next_mobility = np.zeros((next_step, 2))
    #     cur_time = int(cur_time / sec_in_step)
    #     for step in range(next_step):
    #         next_mobility[step][0] = self.mob[(cur_time+step) * sec_in_step][0]
    #         next_mobility[step][1] = self.mob[(cur_time+step) * sec_in_step][1]
    #     return next_mobility

    def get_pre_mobility(self, cur_time, sec_in_step=10):
        next_step = self.pred_mobility_length
        # get next 10 step, while each step is 30 seconds
        pre_mobility = np.zeros((next_step, 2))
        cur_time = int(cur_time / sec_in_step)

        for step in range(next_step):
            if cur_time < next_step:
                pre_mobility[step][0] = self.mob[(cur_time) * sec_in_step][0]
                pre_mobility[step][1] = self.mob[(cur_time) * sec_in_step][1]
            else:
                pre_mobility[step][0] = self.mob[(cur_time-next_step+step) * sec_in_step][0]
                pre_mobility[step][1] = self.mob[(cur_time-next_step+step) * sec_in_step][1]
        return pre_mobility

    def generate_request(self, edge_id):
        """
        Initiate the request
        :param edge_id:
        :return:
        """

        req = Request(self.user_id, edge_id)
        self.req_pool.append(req)
        self.req_count += 1
        return req

    def switch_request(self):
        if len(self.req_pool) > 1:
            self.req_pool.pop(0)
            self.req = self.req_pool[0]


    def finish_request(self, req):
        self.fin_req += 1
        self.req_pool.pop(0)

    def reset_req_count(self, E):
        self.req_count = 0
        self.fin_req = 0
        for req in self.req_pool:
            edge_id = req.edge_id
            E[edge_id].req_pool.remove(req)
            self.req_pool.remove(req)
    def request_update(self, req, time_index):
        self.loc = self.pre_locs[time_index]
        # 0: init and upload->server, 1: server start process, 2: req processed in edge, 3: req finish process, 4: download to user
        # 5: default request.state == 5 means disconnection ,6 means migration
        up_times, process_times, down_times = 0, 0, 0
        up_factor, process_factor, down_factor = 1, 0.2, 1
        # synchronize
        req.u2e_size = req.tasktype.req_u2e_size
        req.process_size = req.tasktype.process_loading
        req.e2u_size = req.tasktype.req_e2u_size


        # if self.req.tasktype.task_size > self.req.resource:
        #     self.req.state = 5
            # print("---------overload---------------")
        # if self.req.state == 5:
        if req.state == 5:
            req.timer += 1


        else:
            # self.req.timer = 0
            rate, dist = trans_rate(self.loc, req.edge_loc)
            # print(f"trans rate:{rate}, dist:{dist}, time:{dist/rate}")
            trans_time = dist/rate
            if req.state == 0: # initial Task
                req.state = 1
                req.u2e_size -= rate  # repeat until u2e_size < 0(finish upload)
                up_times += trans_time

                # 原式应该为: self.req.u2e_size -= trans_rate(self.loc, self.req.edge_loc) * per_second
            # if self.req.state == 1: # Task uploading from user to edge server
            #     if self.req.u2e_size > 0:
            #         self.req.state = 2
            # if self.req.state == 2: # calculate in edge server
            #     process_times += process_rate(self.req.process_size, r_bound)
            #     self.req.state = 3
            if req.state == 3:
                req.e2u_size -= 10000  # value is small,so simplify
                down_times += trans_time
                req.state = 4 if req.e2u_size < 0 else 3# finish task
            # else: # 3, 4, 6
            #     if self.req.e2u_size > 0:
            #         self.req.e2u_size -= 10000  # B*math.log(1+SINR(self.user.loc, self.offloading_serv.loc), 2)/(8*time_scale)
            #         down_times += trans_time
            #     else:
            #         self.req.state = 4 # finish task
        total_time = up_times * up_factor + process_times * process_factor + down_factor * down_times
        req.timer += total_time
        return req

    def mobility_update(self, time):  # t: second
        """
        update the mobility by self.mob
        :param time:
        :return:
        """
        if time < len(self.mob[50:, 0]):
            self.loc[0] = self.mob[time][0]   # x
            self.loc[1] = self.mob[time][1]   # x

            self.pre_locs = self.get_pre_mobility(cur_time=time)
            # self.next_locs = self.get_next_mobility(cur_time=time)
        else:
            self.loc[0] = self.mob[20][0]
            self.loc[0] = self.mob[20][0]

            self.pre_locs = self.get_pre_mobility(cur_time=time)



    def self_process(self, req):
        # print(f"trans rate:{rate}, dist:{dist}, time:{dist/rate}")
        if req.state == 0:  # initial Task
            if self.resource > req.tasktype.process_loading:
                req.timer += process_rate(req.tasktype.process_loading, self.resource, 1)
                self.resource -= req.tasktype.process_loading
                self.resource = max(0, self.resource)
                req.state = 2
                return True
        elif req.state == 2:
            req.energy = proc_energy(100 - self.resource, 100)
            self.resource += req.tasktype.process_loading
            self.resource = min(100, self.resource)
            req.state = 3  # 3: finish in edge
        if req.state == 3:
            req.state = 4 if req.e2u_size < 0 else 3  # finish task
            return True
        return False

    def release(self):
        self.num_step = 0  # the number of step
        self.req_count = 0
        self.fin_req = 0
        self.req_pool = []
        self.resource = self.resource_limit


#############################UE###########################

def trans_rate(user_loc, edge_loc):
    # 计算用户设备（UE）和边缘服务器（MEC）之间的数据传输速率
    B = b_bound # 系统带宽，单位为 Hz（这里为 2 MHz）。决定了通信链路可用的频谱资源范围。
    P = 0.25 # 用户设备的信号强度。发射功率，单位为 W（这里为 0.25 W）。
    d = np.sqrt(np.sum(np.square(user_loc[0] - edge_loc))) + 0.01 # 计算用户设备和边缘服务器之间的欧几里得距离，加上 0.01 避免分母为零（通信距离不能为零）。用于估算信号衰减。
    h = 4.11 * math.pow(300 / (4 * math.pi * 0.15 * d), 2) # 计算用户设备和边缘服务器之间的欧几里得距离，加上 0.01 避免分母为零（通信距离不能为零）。用于估算信号衰减。
    # N = 1e-10 # 表示背景噪声的功率大小.噪声功率密度，单位为 W。
    N = 1e-2 # 表示背景噪声的功率大小.噪声功率密度，单位为 W。

    return B * math.log2(1 + P * h / N), d # 香农公式：用于计算信道的最大传输速率。

def process_rate(req_resource, edge_resource, factor=0.2):
    # server factor:0.2, self-factor=1
    return (req_resource+0.01)/(edge_resource+0.01) * factor #

def proc_energy(idle_resource, total_resrouce):
    P = 0.25 * total_resrouce
    u = 1e-2
    return idle_resource/total_resrouce * P  * u


class Request():
    def __init__(self, user_id, edge_id):
        # id
        self.user_id = user_id
        self.edge_id = edge_id
        self.user_pred_mob = []
        self.edge_loc = 0
        # state
        self.state = 0     # 5: not connect
        self.pre_state = 0
        # transmission size
        self.u2e_size = 0
        self.process_size = 0
        self.e2u_size = 0
        # edge state
        self.resource = 0
        self.energy = 0
        self.mig_size = 0
        # tasktype
        self.tasktype = TaskType()
        self.last_offloading = 0
        # timer
        self.timer = 0

class TaskType():
    def __init__(self):
        ##Objection detection: VOC SSD300
        # transmission
        self.task_type = random.randint(0, 4)

        self.task_size_list = [50, 100, 200, 400, 600]
        self.task_size = self.task_size_list[self.task_type]

        self.req_u2e_size = self.task_size*random.uniform(0.05,0.2) # 表示从用户设备（User）到边缘服务器（Edge）传输的数据大小。
        self.req_e2u_size = self.task_size*random.uniform(0.05,0.2) # 表示任务完成后，边缘服务器需要返回给用户的数据大小。
        self.process_loading = self.task_size - self.req_e2u_size - self.req_u2e_size # 表示任务在边缘服务器上需要的计算量（处理负载）。# 1-400

        # migration
        self.migration_size = 200 #  表示任务迁移时需要传输的上下文数据大小。
    def task_inf(self):
        return "task type:"+str(self.task_type)+" task size:"+str(self.task_size)+"req_u2e_size:" + str(self.req_u2e_size) + "\nprocess_loading:" + str(self.process_loading) + "\nreq_e2u_size:" + str(self.req_e2u_size)


#############################EdgeServer###################

class EdgeServer():
    def __init__(self, edge_id, loc, r_bound=3000):
        self.edge_id = edge_id  # edge server number
        self.loc = loc
        self.capability = r_bound
        self.user_group = []
        self.req_pool = []
        self.overload = 0
        self.limit = LIMIT
        self.connection_num = 0
        self.res_factor = random.uniform(0.5, 1.5)
        self.capability = r_bound * self.res_factor

    def process_request(self, R):
        for req in self.req_pool:
            if req.state == 1:
                if R[self.edge_id]> req.tasktype.process_loading:
                    self.capability = R[self.edge_id]
                    req.timer += process_rate(req.tasktype.process_loading, self.capability, factor=0.2)
                    self.capability -= req.tasktype.process_loading
                    if self.capability <= 0.3 * r_bound:
                        self.overload += 1
                    self.capability = max(0, self.capability)
                    R[self.edge_id] = self.capability
                    req.state = 2
                # else:
                #     print("R[self.edge_id]",R[self.edge_id]," < req.tasktype.process_loading:", req.tasktype.process_loading)
            elif req.state == 2:
                self.capability = R[self.edge_id]
                req.energy = proc_energy(r_bound-self.capability, r_bound)
                self.capability += req.tasktype.process_loading
                # self.capability += req.tasktype.req_u2e_size
                self.capability = min(r_bound, self.capability)
                R[self.edge_id] = self.capability
                req.state = 3  # 3: finish in edge
        return R

    def maintain_request(self, R, U):
        for user in U:
            # the number of the connection user
            self.connection_num = 0
            for user_id in self.user_group:
                if U[user_id].req.state != 6:
                    self.connection_num += 1
            # maintain the request
            if user.req.edge_id == self.edge_id and R[self.edge_id] - user.req.resource > 0:
                # maintain the preliminary connection
                if user.req.user_id not in self.user_group and self.connection_num+1 <= self.limit:
                    # first time : do not belong to any edge(user_group)
                    self.user_group.append(user.user_id)  # add to the user_group
                    user.req.state = 0  # prepare to connect
                    # notify the request
                    user.req.edge_id = self.edge_id
                    user.req.edge_loc = self.loc

                # update the resource
                self.capability -= user.req.resource
                R[self.edge_id] = self.capability


    def migration_update(self, O, B, table, U, E):
        migration_timer_factor = 0.01
        migration_timer = 0
        # maintain the migration
        for user_id in self.user_group:
            # prepare to migration
            if U[user_id].req.edge_id != O[user_id]: # 如果当前用户的请求（req）指定的边缘服务器（edge_id）与目标边缘服务器（O[user_id]）不同，则说明需要进行任务迁移
                # initial
                ini_edge = int(U[user_id].req.edge_id)
                target_edge = int(O[user_id])
                if table[ini_edge][target_edge] - B[user_id] >= 0:
                    # on the way to migration, but offloading to another edge computer(step 0)
                    if U[user_id].req.state == 6 and target_edge != U[user_id].req.last_offloading:
                        # reduce the bandwidth
                        table[ini_edge][target_edge] -= B[user_id]
                        # start migration
                        U[user_id].req.mig_size = U[user_id].req.tasktype.migration_size
                        U[user_id].req.mig_size -= B[user_id]
                        migration_timer += 1
                        # print("user", U[user_id].req.user_id, ":migration step 0")
                    # first try to migration(step 1)
                    elif U[user_id].req.state != 6:
                        table[ini_edge][target_edge] -= B[user_id]
                        # start migration
                        U[user_id].req.mig_size = U[user_id].req.tasktype.migration_size
                        U[user_id].req.mig_size -= B[user_id]
                        # store the pre state
                        U[user_id].req.pre_state = U[user_id].req.state
                        # on the way to migration, disconnect to the old edge
                        U[user_id].req.state = 6
                        migration_timer += 1
                        # print("user", U[user_id].req.user_id, ":migration step 1")
                    elif U[user_id].req.state == 6 and target_edge == U[user_id].req.last_offloading:
                        # keep migration(step 2)
                        if U[user_id].req.mig_size > 0:
                            # reduce the bandwidth
                            table[ini_edge][target_edge] -= B[user_id]
                            U[user_id].req.mig_size -= B[user_id]
                            migration_timer += 1
                            # print("user", U[user_id].req.user_id, ":migration step 2")
                        # end the migration(step 2)
                        else:
                            # the number of the connection user
                            target_connection_num = 0
                            for target_user_id in E[target_edge].user_group:
                                if U[target_user_id].req.state != 6:
                                    target_connection_num += 1
                            # print("user", U[user_id].req.user_id, ":migration step 3") # finish migration
                            # change to another edge
                            if E[target_edge].capability - U[user_id].req.resource >= 0 and target_connection_num + 1 <= E[target_edge].limit:
                                # register in the new edge
                                E[target_edge].capability -= U[user_id].req.resource
                                E[target_edge].user_group.append(user_id)
                                self.user_group.remove(user_id)
                                # update the request
                                # id
                                U[user_id].req.edge_id = E[target_edge].edge_id
                                U[user_id].req.edge_loc = E[target_edge].loc
                                # release the pre-state, continue to transmission process
                                U[user_id].req.state = U[user_id].req.pre_state
                                print("user", U[user_id].req.user_id, ":migration finish")
            #store pre_offloading
            U[user_id].req.last_offloading = int(O[user_id])
            migration_timer *= migration_timer_factor

        return table, migration_timer

    #release the all resource
    def release(self):
        # factor = random.uniform(0.5, 1.5)
        self.capability = r_bound * self.res_factor
        self.user_group = []
        self.req_pool = []
        self.overload = 0
        self.limit = LIMIT
        self.connection_num = 0
