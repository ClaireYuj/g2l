import gc

import math

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from tqdm import tqdm
from torch.distributions import Normal

from algorithm.GCN import GraphTransformer, GatedGT
import algorithm.GCN as GCN
from utils.util import get_log_prob
from utils.util import init

from utils.util import normalization
from src.ReplayBuffer import PrioritizedReplayBuffer

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#####################  hyper parameters  ####################
LR_A = 0.0001  # learning rate for actor
LR_C = 0.0001  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
BATCH_SIZE = 1
OUTPUT_GRAPH = False


class IPPO(object):
    def __init__(self, s_dim, r_dim, b_dim, o_dim, e_dim, u_dim, mobi_dim, r_bound, b_bound, n_candidate_edge,
                 memory_capacity,
                 n_hidden=512, tau=0.001, lr_a=1e-4, lr_c=1e-4, batch_size=64, graph_mode='e2e'):  # graph mode:e2e, u2e
        self.seq_len = int(mobi_dim)
        self.memory = PrioritizedReplayBuffer(max_size=memory_capacity, state_dim=s_dim, action_dim=e_dim,
                                              n_agent=u_dim, seq_len=mobi_dim, mode='leader')  # a_dim=r_dim+o_dim

        self.n_agent = u_dim
        self.batch_size = batch_size
        self.n_candidate_edge = n_candidate_edge  # action shaping
        self.category = 0  # 0:UE, 1:MEC
        self.s_dim = s_dim
        # self.a_dim = r_dim + b_dim + o_dim #1700
        self.a_dim = e_dim
        self.r_dim = r_dim
        self.e_dim = e_dim
        self.b_dim = b_dim
        self.o_dim = o_dim
        self.r_bound = r_bound
        self.b_bound = b_bound
        # self.memory = np.zeros((self.memory_capacity, s_dim * 2 + self.a_dim + 1), dtype=np.float32)  # s_dim + a_dim + r + s_dim
        # self.memory = torch.zeros((self.memory_capacity, s_dim * 2 + self.a_dim + 1), dtype=torch.float32) # s_dim + a_dim + r + s_dim
        # self.memory = ReplayBuffer(self.memory_capacity, self.s_dim, self.r_dim+self.o_dim, self.n_agent, self.seq_len) # a_dim=r_dim+o_dim
        self.tau = tau  # Soft update parameter
        # Initialize memory (replay buffer)
        self.pointer = 0

        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Initialize networks
        self.n_hidden = n_hidden
        print("Initialize networks")

        self.actor_max_norm = 0.5
        self.critic_max_norm = 1.0
        self.discount = 0.99
        self.bc_coef = 2.5  # 行为克隆权重

        print("Initialize target networks")

        self.actor_critic = ActorCritic(s_dim, self.r_dim, self.e_dim, self.n_agent, n_hidden=n_hidden).to(self.device)
        self.ac_target = deepcopy(self.actor_critic)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr_a)
        # self.optimizer = optim.Adam(self.actor.parameters(), lr=lr_a)


    def choose_actions(self, s, algo_name='ppo', is_target=False, ):
        """
        根据当前的状态 s 选择一个动作 a。
        :param s: 当前状态
        :return: 选择的动作
        """
        with torch.no_grad():
            out, edge, val = self.actor_critic(s)
        return out, edge, val
        # s = s.clone().detach().to(self.device).float()    # 转换为 PyTorch 张量
        # with torch.no_grad():  # 不计算梯度
        #     # r_out, b_out, offload_out = self.actor(s)  # 通过 Actor 网络获取动作的输出
        #     mec_out = self.actor(s)
        #     # 你可以根据具体情况，选择合适的动作（例如使用 softmax 或其他方法）
        #     # return torch.cat([r_out, b_out, offload_out], dim=-1) # 返回动作
        #     return mec_out

    def save_model(self, path='ac_model.pth', target_critic=None):
        """
        保存 Actor-Critic 模型及其优化器状态。

        参数:
        - actor: Actor 网络模型。
        - critic: Critic 网络模型。
        - actor_optimizer: Actor 的优化器。
        - critic_optimizer: Critic 的优化器。
        - target_critic: (可选) 目标 Critic 网络。
        - path: 保存文件路径。
        - step_count: (可选) 当前训练的步数。
        """
        save_data = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_critic_optimizer_state_dict': self.optimizer.state_dict()
        }

        if target_critic:
            save_data['target_critic_state_dict'] = target_critic.state_dict()

        torch.save(save_data, path)
        print(f"save model to: {path}")

    def load_model(self, path='ac_model.pth', target_critic=None):

        checkpoint = torch.load(path)

        self.actor_critic = ActorCritic(self.s_dim, self.r_dim, self.e_dim, self.n_agent, n_hidden=self.n_hidden).to(
            self.device)

        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])

        self.optimizer.load_state_dict(checkpoint['actor_critic_optimizer_state_dict'])
        if target_critic and 'target_critic_state_dict' in checkpoint:
            target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        print(f"load mode from {path} ")

    def release(self):
        del self.actor
        del self.critic
        del self.actor_critic
        gc.collect()

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update of target network parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_((tau * local_param.data) + ((1.0 - tau) * target_param.data))

    def action_shaping(self, a):
        # only store the choice with highest proability
        # cut_a =  a[:,:, self.r_dim:]
        cut_a = a
        topk = self.n_candidate_edge

        n_agent, seq_len, state = cut_a.shape
        shaping_a = torch.zeros((n_agent, seq_len, topk))
        num = 0
        for idx in range(n_agent - 1):
            # agent_act = cut_a[idx, :, idx*self.a_dim:(idx+1)*self.a_dim]
            agent_act = cut_a[idx, :, :]
            # probabilities = F.softmax(agent_act, dim=-1)
            # actions = torch.multinomial(probabilities, num_samples=topk)
            values, actions = torch.topk(agent_act, topk)

            shaping_a[idx, :, :topk] = actions
        return shaping_a

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)



    def train(self, batch_size=1, gamma=0.99, gae_lambda=0.95, algo_name='ppo'):
        # indices = np.random.choice(self.memory_capacity, size=BATCH_SIZE)

        # indices = np.random.choice(self.memory_capacity, size=BATCH_SIZE) # generate data to train
        # bt = self.memory[indices, :]
        batch_id, batch_s, batch_a, batch_r, batch_s_, batch_log, batch_val, bter = self.memory.sample_buffer(
            batch_size)
        # self.actor_critic.train()
        # self.critic.train()
        # for i in tqdm(range(batch_size), desc=f"training:", ncols=100, ascii=True):
        total_loss = 0
        for i in range(batch_size):
            # bs = bs.permute(1,0,2,3).flatten(1, 2).clone() #[batch, n_agent, seq, action] -> [batch*n_agent, seq, action]
            # ba = ba.permute(1,0,2,3).flatten(1, 2).clone()
            # bs_ = bs_.permute(1,0,2,3).flatten(1, 2).clone()
            bs = batch_s[i, :, :, :]
            ba = batch_a[i, :, :, :]
            bs_ = batch_s_[i, :, :, :]
            br = batch_r[i]
            q_values = batch_val[i, :, :, :].clone().detach()
            # ba_log = batch_log[i]
            ba_log = get_log_prob(ba)
            # torch.autograd.set_detect_anomaly(True)
            done = bter[i, :]
            # _, q_values = self.choose_action(bs,algo_name=algo_name)
            std = torch.exp(ba_log)
            dist = Normal(ba, std)
            entropy = dist.entropy().sum(dim=-1).mean()

            # Critic network update
            # with torch.no_grad():
            _, next_actions, next_q_values = self.actor_critic(bs_)

            # next_actions, _, next_q_values = self.choose_actions(bs_, algo_name=algo_name, is_target=False)
            next_actions_log = get_log_prob(next_actions)
            # next_actions, next_actions_log = self.choose_action(bs_, algo_name=algo_name,is_target=False)
            # next_q_values = self.critic_target(bs_, next_actions)
            target_q_values = br + gamma * next_q_values  # TD
            # advantage = torch.zeros_like(next_q_values, dtype=torch.float32)
            advantage = torch.zeros_like(q_values, dtype=torch.float32)
            gae = 0
            for step in reversed(range(q_values.shape[1])):
                if step == self.seq_len - 1:
                    delta = br[:, step] - q_values[:, step]

                    # delta = br  + gamma * q_values[step] - next_q_values[step]
                else:
                    delta = br[:, step] + gamma * q_values[:, step + 1] - q_values[:, step]

                gae = delta + gamma * gae_lambda * gae  # gae_lambda=0.9 使用lambda平滑GAE
                advantage[:, step] = gae
            ratio = torch.exp(next_actions_log - ba_log) + 1e-8  # 比率 r_t(θ)
            # ratio = torch.exp(next_actions_log - ba_log)  # 比率 r_t(θ)
            # Compute the clipped loss for PPO
            epsilon = 0.2
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            # advantage = advantage.squeeze()
            actor_loss = -torch.mean(torch.min(ratio * advantage, clipped_ratio * advantage))  # policy_loss
            # critic_loss = torch.mean((target_q_values-q_values) ** 2) # value_loss
            critic_loss = F.mse_loss(target_q_values, q_values)
            self.memory.update_priorities(batch_id[i], critic_loss)

            loss = actor_loss + critic_loss - entropy * 0.01
            total_loss = total_loss + loss

        # before_update = torch.cat([param.view(-1) for param in self.actor_critic.parameters()])
        self.optimizer.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_max_norm)
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_max_norm)
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.actor_max_norm)
        self.optimizer.step()
        # after_update = torch.cat([param.view(-1) for param in self.actor_critic.parameters()])
        # print("\n===================Leader model para========================")
        # print((after_update - before_update).norm())

        self.soft_update(self.actor_critic, self.ac_target, self.tau)
        # self.actor_optimizer.zero_grad()
        # actor_loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_max_norm)
        # self.actor_optimizer.step()
        #
        # ## critic update
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward() # actor share weights?
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_max_norm)
        # self.critic_optimizer.step()

        # self.memory.clear()
        return loss


class ActorNetwork(nn.Module):
    """
    A network for actor
    """




    def __init__(self, state_dim, e_dim, output_size, output_activation=torch.sigmoid, init_w=3e-3):
        super(ActorNetwork, self).__init__()
        self.e_dim = e_dim
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, e_dim)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        # activation function for the output
        self.output_activation = output_activation
        self.log_std = nn.Parameter(torch.zeros(1, e_dim))


    def forward(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        if self.output_activation == nn.functional.softmax:
            out = self.output_activation(self.fc3(out), dim=-1)
        else:
            out = self.output_activation(self.fc3(out))

        probs = F.softmax(out)
        act = torch.argmax(probs, dim=-1).unsqueeze(-1)
        return out, act

    def get_distribution(self, x):
        action_mean = self.forward(x)
        return torch.distributions.Normal(action_mean, self.log_std.exp())

class CriticNetwork(nn.Module):
    """
    A network for critic
    """

    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, global_state):
        return self.net(global_state)



class ActorCritic(nn.Module):
    def __init__(self, s_dim, r_dim, e_dim, n_agent, mobi_dim=2, device=torch.device("cuda:0"), n_hidden=200,
                 num_heads=1):
        super(ActorCritic, self).__init__()
        self.r_dim = r_dim
        self.s_dim = s_dim
        self.actor = ActorNetwork(state_dim=s_dim, e_dim=e_dim, output_size=1)
        self.critic = CriticNetwork(s_dim, 1)
        self.device = device

    def forward(self, s):
        n_agent, seq, f = s.shape

        # s = torch.cat([r, offload], dim=-1)
        action_out, action = self.actor(s)
        # action_out = action.repeat(1, seq, 1)
        value = self.critic(s)
        return action_out, action, value
