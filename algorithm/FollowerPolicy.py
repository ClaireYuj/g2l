import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from algorithm.LighterFollower import LighterFollower
# from algorithm.MultiFollower import MultiFollower, ActorCritic
import os
import torch.nn as nn
import torch_geometric.nn as geo_nn
import algorithm.GCN as GCN

from src.ReplayBuffer import  PrioritizedReplayBuffer
from utils.util import get_log_prob
from utils.util import normalization
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class FollowerPolicy():

    def __init__(self, state_dim, action_dim, n_agent, seq_len, e_dim,obs_dim, memory_capacity, candidate_choice=1,batch_size=4,device=torch.device("cuda:0"),
                 hidden_dim=512, n_head=1, add_state_token=True,shared_obs_dim=3, lr=1e-4, gamma=0.99, clip_eps=0.2, tau=0.001):
        self.device = device
        self.memory = PrioritizedReplayBuffer(max_size=memory_capacity, state_dim=state_dim+obs_dim, action_dim=1, n_agent=n_agent, seq_len=seq_len, mode='follower') #n_candidate+2表示候选服务器和 loc
        self.seq_len = seq_len
        self.num_agents = n_agent
        self.e_dim=e_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.action_dim = action_dim
        self.clip_eps = clip_eps
        self.obs_dim=obs_dim
        self.candidate_choice = candidate_choice
        self.shared_obs_dim = shared_obs_dim # n_candidate_edge
        self._use_policy_active_masks = True
        self.batch_size = 1
        self.actor_max_norm = 0.5
        self.critic_max_norm = 1.0
        self.bc_coef = 2.5
        self.actor = TGCNActor(state_dim=state_dim+obs_dim, action_dim=action_dim, e_dim=e_dim, n_agent=n_agent, seq_len=seq_len, num_layers=2, pred_len=5, hidden_dim=hidden_dim).to(
            self.device)  # obs_dim: candidate edge state
        self.actor_target = TGCNActor(state_dim=state_dim+obs_dim, action_dim=action_dim, e_dim=e_dim,n_agent=n_agent,
                                            seq_len=seq_len, num_layers=2, pred_len=5, hidden_dim=hidden_dim).to(self.device)

        # self.actor = LighterFollower(state_dim=state_dim, action_dim=action_dim, n_agent=n_agent, seq_len=seq_len, obs_dim=obs_dim, n_head=n_head, candidate_choice=candidate_choice).to(self.device) #obs_dim: candidate edge state
        # self.actor_target = LighterFollower(state_dim=state_dim, action_dim=action_dim, n_agent=n_agent, seq_len=seq_len, n_head=n_head, obs_dim=obs_dim, candidate_choice=candidate_choice).to(self.device)
        # self.critic = ActorCritic(n_head=1, n_agent=n_agent, action_dim=action_dim, n_block=1, device=device, n_embed=1).to(self.device)
        # self.critic_target = ActorCritic(n_head=1, n_agent=n_agent, action_dim=action_dim, n_block=1, device=device, n_embed=1).to(self.device)
        # critic:
        self.critic = Critic(state_dim=state_dim+obs_dim, action_dim=1, n_heads=n_head).to(self.device)
        self.critic_target = Critic(state_dim=state_dim+obs_dim, action_dim=1, n_heads=n_head).to(self.device)

        self.actor_critic = ActorCritic(state_dim=state_dim, action_dim=action_dim, e_dim=e_dim,n_agent=n_agent, seq_len=seq_len, obs_dim=obs_dim, n_head=n_head, hidden_dim=hidden_dim,candidate_choice=candidate_choice).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.tau = tau
        self.critic_loss_fn = torch.nn.MSELoss()
        self.device = device
        # self.adj = GCN.construct_follower_matrix(self.candidate_choice, n_agent).to(self.device)
        # self.adj = torch.eye(self.candidate_choice).to(self.device)
        self.adj = torch.eye(hidden_dim).to(self.device)

    def choose_actions(self, state):
        n_agent, seq_len, f = state.shape
        actor_action = torch.zeros((self.n_agents, seq_len, self.action_dim))

        # if algo_name == "td3":
        #     state = torch.cat([state, obs], dim=-1).to(self.device)
        #
        #     # v_loc, output_action, output_action_log = self.actor(state, self.adj)
        #     output_action = self.actor(state, self.adj)
        #
        #     # state = torch.cat([state, obs], dim=-1)
        #     val = self.critic(state, output_action)
        # if algo_name == "td3":
        #     v_loc, output_action, output_action_log = self.actor(state, obs, n_agent)
        #     state = torch.cat([state, obs], dim=-1)
        #     val = self.critic(state, output_action)

        for agent_id in range(self.n_agents):
            output_action = self.actor_critic(state)
            actor_action[agent_id, :, :] = output_action
        return output_action
        # return self.actor(state, obs, n_agent)


    def compute_advantages(self, rewards, values, dones, next_values):
        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        return advantages, returns

    def learn_td3(self, gamma=0.99,  gae_lambda = 0.9):
        # states, actions, old_log_probs, returns, advantages
        # batch_s, batch_a, batch_r, batch_s_, batch_log, bter = self.memory.sample_buffer(self.batch_size)
        batch_id,batch_s, batch_a, batch_r, batch_s_, batch_log, batch_val, bter = self.memory.sample_buffer(self.batch_size)

        self.actor.train()
        self.critic.train()

        # 初始化累计 Loss
        total_actor_loss = 0
        total_critic_loss = 0

        # # for i in tqdm(range(self.batch_size), desc=f"training:", ncols=100, ascii=True):
        for i in range(self.batch_size):

            # bs = bs.permute(1,0,2,3).flatten(1, 2).clone() #[batch, n_agent, seq, action] -> [batch*n_agent, seq, action]
            # ba = ba.permute(1,0,2,3).flatten(1, 2).clone()
            # bs_ = bs_.permute(1,0,2,3).flatten(1, 2).clone()
            bs = batch_s[i,:,:,:]
            ba = batch_a[i,:,:,:]
            bs_ = batch_s_[i, :, :, :]
            br = batch_r[i]
            ba_log = batch_log[i,:, :, :]
            n_agent, _, _ = bs_.shape
            act_s = bs[:, :, :-self.obs_dim]
            act_obs  = bs[:, :, -self.obs_dim:]
            act_s_ = bs_[:, :, :-self.obs_dim]
            act_obs_ = bs_[:, :, -self.obs_dim:]
            q_values = self.critic(bs, ba)

            # print("update networks")
            with torch.no_grad():
                state_ = torch.cat([act_s_, act_obs_], dim=-1)
                # val, next_actions, next_actions_log = self.actor_target(act_s_, act_obs_, n_agent)
                next_actions = self.actor_target(state_, self.adj)

                # val, next_actions, next_actions_log = self.actor_target(act_s_, act_obs_, ba, n_agent)
                next_actions = next_actions.to(self.device)
                next_q_values = self.critic_target(bs_, next_actions)
                target_q_values = br + gamma * next_q_values
            ## finish torch.no_grad

            ## critic update
            critic_loss = torch.mean((q_values - target_q_values) ** 2)
            total_critic_loss = total_critic_loss + critic_loss
            # actor_loss = torch.tensor(0,dtype=torch.float32)

            ## actor update
            # if np.random.randint(0, 2) == 0:
            # val, bc_action, _ = self.actor(act_s, act_obs, n_agent)
            state = torch.cat([act_s, act_obs], dim=-1)
            bc_action = self.actor(state, self.adj)

            bc_action = bc_action.to(self.device)
            bc_loss = nn.MSELoss()(bc_action.float(), ba.float())
            # before_update = [param.clone() for param in self.actor.parameters()]
            actor_loss = -self.critic(bs, bc_action).mean() + self.bc_coef * bc_loss
            total_actor_loss = total_actor_loss + actor_loss
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_max_norm)
        self.actor_optimizer.step()
        self.soft_update(self.actor, self.actor_target, self.tau)

        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_max_norm)
        self.critic_optimizer.step()



        loss=total_actor_loss+0.5*total_critic_loss
            # print("*****************************model para******************************")
            # after_update = [param.clone() for param in self.actor.parameters()]
            # for b, a in zip(before_update, after_update):
            #     print((b - a).norm())  # 输出非零说明参数更新了
            # Perform soft update of target networks
        self.soft_update(self.actor, self.actor_target, 0.001)
        self.soft_update(self.critic, self.critic_target, 0.001)
            # self.memory.clear()
        return loss

    def learn_ppo(self, gamma=0.99,  gae_lambda = 0.9, algo_name='ppo'):
        # states, actions, old_log_probs, returns, advantages
        # bs, ba, br, bs_, b_log, bter = self.memory.sample_buffer(self.batch_size)
        #
        # bs = bs.permute(1,2,0,3).flatten(1,2).to(self.device)  #[batch, agent, seq, state]-> [agent, seq, batch,state]->[n_agent,batch*seq,state]
        # ba = ba.permute(1,2,0,3).flatten(1,2).to(self.device)
        # bs_ = bs_.permute(1,2,0,3).flatten(1,2).to(self.device)
        # ba_log = get_log_prob(ba)
        batch_id, batch_s, batch_a, batch_r, batch_s_, batch_log, batch_val, bter = self.memory.sample_buffer(self.batch_size)

        total_loss = 0
        for i in range(self.batch_size):

            bs = batch_s[i,:,:,:]
            ba = batch_a[i,:,:,:]
            bs_ = batch_s_[i, :, :, :]
            br = batch_r[i, :, :, :]
            ba_log = batch_log[i,:, :, :]
            n_agent, _, _ = bs_.shape
            q_values = batch_val[i, :, :, :].clone().detach()
            done = bter[i,:]


            act_s_ = bs_[:, :, :-self.obs_dim]
            act_obs_ = bs_[:, :, -self.obs_dim:]
            # q_values = batch_val[i,:,:,:]
            # print("update networks")
            # with torch.no_grad():
            next_actions, next_q_values = self.choose_actions(act_s_, act_obs_, n_agent, algo_name=algo_name)
            next_actions_log = get_log_prob(next_actions)
                # val, next_actions, next_actions_log = self.actor_target(act_s_, act_obs_, n_agent)
                # next_actions = next_actions.to(self.device)
                # next_actions_log = next_actions_log.to(self.device)
                # next_q_values = self.critic_target(bs_, next_actions)
            target_q_values = br + gamma * next_q_values
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
                ## finish torch.no_grad
            # ratio = torch.exp(next_actions_log-b_log)  # 比率 r_t(θ)
            ratio = torch.exp(next_actions_log-ba_log)  # 比率 r_t(θ)

            # Compute the clipped loss for PPO
            epsilon = 0.2
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            actor_loss = -torch.mean(torch.min(ratio * advantage, clipped_ratio * advantage))  # 裁剪的目标函数
            # critic_loss = torch.mean((target_q_values - q_values) ** 2)
            critic_loss = F.mse_loss(target_q_values, q_values)
            self.memory.update_priorities(batch_id[i], critic_loss)

            loss = actor_loss + critic_loss
            total_loss = total_loss + loss
        # before_update = torch.cat([param.view(-1) for param in self.actor_critic.parameters()])

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_max_norm)
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.optimizer.step()
        # after_update = torch.cat([param.view(-1) for param in self.actor_critic.parameters()])
        # print("\n===================follower model para========================")
        # print((after_update - before_update).norm())
        self.soft_update(self.actor, self.actor_target, 0.001)
        self.soft_update(self.critic, self.critic_target, 0.001)
        return loss

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update of target network parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_((tau * local_param.data) + ((1.0 - tau) * target_param.data))

    def save_model(self, path='marl_model.pth'):
        save_data = {
            'actors': [actor.state_dict() for actor in self.actor],
            'critics': [critic.state_dict() for critic in self.critic],
            'actor_critic': [actor_critic.state_dict() for actor_critic in self.actor_critic],
            'actor_optimizers': [actor_opt.state_dict() for actor_opt in self.actor_optimizer],
            'critic_optimizers': [critic_opt.state_dict() for critic_opt in self.critic_optimizer],
            'actor_critic_optimizers': [opt.state_dict() for opt in self.optimizer]

        }
        torch.save(save_data, path)
        print(f"save model to : {path}")

    def load_model(self,path='marl_model.pth'):
        checkpoint = torch.load(path)
        for actor, state_dict in zip(self.actors, checkpoint['actors']):
            actor.load_state_dict(state_dict)
        for actor_opt, state_dict in zip(self.actors_optimizer, checkpoint['actor_optimizers']):
            actor_opt.load_state_dict(state_dict)
        for ac, state_dict in zip(self.actor_critic, checkpoint['actor_critic']):
            ac.load_state_dict(state_dict)
        for opt, state_dict in zip(self.optimizer, checkpoint['actor_critic_optimizers']):
            opt.load_state_dict(state_dict)
        for critic, state_dict in zip(self.critics, checkpoint['critics']):
            critic.load_state_dict(state_dict)
        for critic_opt, state_dict in zip(self.critics_optimizer, checkpoint['critic_optimizers']):
            critic_opt.load_state_dict(state_dict)
        print(f"load model from {path} ")




class TGCNActor(nn.Module):
    def __init__(self, state_dim, hidden_dim, e_dim,num_layers, action_dim, n_agent, seq_len, pred_len, device=torch.device("cuda:0"),):
        """
        TGCN 模型
        :param input_dim: 输入特征维度
        :param hidden_dim: 隐藏状态维度
        :param num_layers: 层数
        :param output_dim: 输出特征维度
        """
        super(TGCNActor, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.e_dim = e_dim
        self.tgcn_cells = nn.ModuleList([TGCNCell(state_dim, hidden_dim) for i in range(num_layers)])

        # self.tgcn_cells = nn.ModuleList([TGCNCell(state_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_dim, e_dim)


    def forward(self, x, adj):
        """
        前向传播
        :param x: 输入数据 (batch_size, N, seq_len,input_dim)
        :param adj: 邻接矩阵 (N, N)
        :return: 输出预测 (batch_size,  N, seq_len,output_dim)
        """
        if x.dim()==4:
            x = x.flatten(0, 1)
        N, seq_len, _ = x.size()
        h = torch.zeros((N, self.hidden_dim), device=x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # 当前时间步的输入 [batch,  agent, seq, feature]
            for i, cell in enumerate(self.tgcn_cells):
                h = cell(x_t, h, adj)
            h_out = self.fc(h)
            probs = F.softmax(h_out)
            h_act = torch.argmax(probs, dim=1).unsqueeze(1)
            outputs.append(h_act)  # 添加当前时间步的输出
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.unsqueeze(-1)
            # outputs.append(self.fc(h).unsqueeze(1))  # 添加当前时间步的输出
        return outputs

class TGCNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        TGCN 单元
        :param input_dim: 输入特征维度
        :param hidden_dim: 隐藏状态维度
        """
        super(TGCNCell, self).__init__()
        self.hidden_dim = hidden_dim
        # self.gcn_z = geo_nn.GCNConv(input_dim + hidden_dim, hidden_dim)
        # self.gcn_r = geo_nn.GCNConv(input_dim + hidden_dim, hidden_dim)
        # self.gcn_h = geo_nn.GCNConv(input_dim + hidden_dim, hidden_dim)
        self.gcn_z = TGCNGraphConvolution(input_dim + hidden_dim, hidden_dim)
        self.gcn_r = TGCNGraphConvolution(input_dim + hidden_dim, hidden_dim)
        self.gcn_h = TGCNGraphConvolution(input_dim + hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x, hidden, adj):
        """
        前向传播
        :param x: 当前时间步的输入 (N, input_dim)
        :param hidden: 上一时间步的隐藏状态 (N, hidden_dim)
        :param adj: 邻接矩阵 (N, N)
        :return: 当前时间步的隐藏状态
        """
        combined = torch.cat([x, hidden], dim=-1)  # (N, input_dim + hidden_dim)

        # 更新门
        z = self.sigmoid(self.gcn_z(combined, adj))
        # 重置门
        r = self.sigmoid(self.gcn_r(combined, adj))
        # 候选隐藏状态
        h_tilde = self.tanh(self.gcn_h(torch.cat([x, r * hidden], dim=-1), adj))
        # 最终隐藏状态
        h_next = z * hidden + (1 - z) * h_tilde
        h_next = self.sigmoid(h_next)
        return h_next


class TGCNGraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        TGCN 图卷积层
        :param input_dim: 输入特征维度
        :param output_dim: 输出特征维度
        """
        super(TGCNGraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        前向传播
        :param x: 输入特征矩阵 (N, input_dim)
        :param adj: 邻接矩阵 (N, N)
        :return: 输出特征矩阵 (N, output_dim)
        """
        support = torch.mm(x, self.weight)  # (N, output_dim)
        # output = torch.mm(adj, support)    # 图卷积操作
        output = torch.mm(support, adj)    # 图卷积操作

        return output + self.bias

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=100, n_heads=2, n_layers=2, device=torch.device("cuda:0")):
        super(Critic, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim

        # transformer_encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=state_dim + action_dim,  # The input dimension (combined state and action dimensions)
        #     nhead=n_heads,  # Number of attention heads
        #     dim_feedforward=hidden_dim  # Hidden dimension size in the feedforward network
        # )
        self.layer = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.device = device
        # Now, define the transformer encoder using the layer
        # self.transformer = nn.TransformerEncoder(
        #     transformer_encoder_layer,  # Pass the encoder layer here
        #     num_layers=n_layers  # Number of layers in the transformer encoder
        # )
        self.fc = nn.Linear(state_dim+action_dim, 1)

    def forward(self, state_sequence, action_sequence):
        state_sequence = state_sequence.to(self.device)
        action_sequence = action_sequence.to(self.device)
        combined = torch.cat([state_sequence, action_sequence], dim=-1)  # Combine state and action
        # x = self.transformer(combined)
        x = self.layer(combined)
        # x = self.fc(x[-1])  # Use the last output from the transformer for value prediction
        return x  # Output Q-value

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agent, e_dim ,seq_len, obs_dim, n_head, candidate_choice,hidden_dim, device=torch.device("cuda:0")):
        super(ActorCritic, self).__init__()
        self.e_dim=e_dim
        self.actor = TGCNActor(state_dim=state_dim+obs_dim, action_dim=action_dim, e_dim=e_dim,n_agent=n_agent, seq_len=seq_len, num_layers=2, pred_len=5, hidden_dim=hidden_dim)
        # self.actor = LighterFollower(state_dim=state_dim, action_dim=action_dim, n_agent=n_agent, seq_len=seq_len, obs_dim=obs_dim, n_head=n_head, candidate_choice=candidate_choice)
        self.critic = Critic(state_dim=state_dim+obs_dim, action_dim=action_dim)
        self.device = device


    def forward(self, state, obs, n_agent, adj):
        s = torch.cat([state, obs], dim=-1).to(self.device)
        output_action = self.actor(s, adj)
        # s = torch.cat([state, obs], dim=-1).to(self.device)
        output_action = output_action.to(self.device)
        val = self.critic(s, output_action)
        return output_action, val