import gc
from thop import profile
import time

from utils.util import get_log_prob

import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
import numpy as np
import random
from copy import deepcopy
from numpy import savetxt
from numpy import loadtxt

from src.ReplayBuffer import PrioritizedReplayBuffer
from env.MP_HRL_Env import Env as Env
import torch.nn.functional as F
from mamba_ssm import Mamba


class MambaFollower(object):
    def __init__(self,n_agents, state_dim, action_dim, seq_len, action_lower_bound, e_dim,
                 action_higher_bound,memory_capacity=10000,
                 InfdexofResult=0,
                 target_tau=1, reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_output_activation=torch.sigmoid, actor_lr=0.0001, critic_lr=0.0001,
                 optimizer_type="adam", max_grad_norm=None, batch_size=6, episodes_before_train=64,
                 epsilon_start=1, epsilon_end=0.01, epsilon_decay=None, use_cuda=True, Benchmarks_mode=None):
        super(MambaFollower, self).__init__()
        self.n_agents = n_agents
        self.e_dim = e_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lower_bound = action_lower_bound
        self.action_higher_bound = action_higher_bound
        self.n_episodes = 0
        self.seq_len = seq_len

        self.memory = PrioritizedReplayBuffer(memory_capacity,self.state_dim, self.action_dim, self.n_agents, self.seq_len, mode='follower')
        self.actor_output_activation = actor_output_activation
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        # params for epsilon greedy
        self.device = torch.device("cuda:0")
        self.reward_scale = 1.
        self.reward_gamma = 0.95

        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.target_tau = target_tau
        self.actors = [MambaActor(self.state_dim, self.e_dim, self.action_dim, self.actor_output_activation).to(self.device)] * self.n_agents

        # critic_state_dim = self.n_agents * self.state_dim
        # critic_action_dim = self.n_agents * self.action_dim
        self.critic_state_dim = self.state_dim
        self.critic_action_dim = self.action_dim
        # self.critics = [LSTMCritic(critic_state_dim, critic_action_dim).to(self.device)] * 1
        self.critics = [CriticNetwork(self.critic_state_dim, self.critic_action_dim).to(self.device)] * 1

        # to ensure target network and learning network has the same weights
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)
        if optimizer_type == "adam":
            self.actors_optimizer = [Adam(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critics_optimizer = [Adam(c.parameters(), lr=self.critic_lr) for c in self.critics]
        elif optimizer_type == "rmsprop":
            self.actors_optimizer = [RMSprop(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critics_optimizer = [RMSprop(c.parameters(), lr=self.critic_lr) for c in self.critics]
        # if self.use_cuda:
        #     for i in range(self.n_agents):
        #         self.actors[i].cuda()
        #         self.critics[i].cuda()
        #         self.actors_target[i].cuda()
        #         self.critics_target[i].cuda()
        self.eval_episode_rewards = []
        self.server_episode_constraint_exceeds = []
        self.energy_episode_constraint_exceeds = []
        self.time_episode_constraint_exceeds = []
        self.eval_step_rewards = []
        self.mean_rewards = []

        self.episodes = []
        self.Training_episodes = []

        self.Training_episode_rewards = []
        self.Training_step_rewards = []

        self.InfdexofResult = InfdexofResult
        # self.save_models('./checkpoint/Benchmark_'+str(self.Benchmarks_mode)+'_checkpoint'+str(self.InfdexofResult)+'.pth')
        self.results = []
        self.Training_results = []
        self.serverconstraints = []
        self.energyconstraints = []
        self.timeconstraints = []


    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)


    # train on a sample batch
    def train(self):
        # do not train until exploration is enough

        tryfetch = 0
        idxs, b_states_var, b_actor_actions_var, b_rewards_var, b_next_states_var, batch_log, batch_val, b_dones_var = self.memory.sample_buffer(
            self.batch_size)

        critic_loss_list = []
        errors = torch.zeros((self.batch_size)).to(self.device)
        loss = []
        whole_states_var = b_states_var.view(-1, self.seq_len, self.n_agents * self.state_dim)
        states_var = whole_states_var.view(-1, self.seq_len, self.n_agents, self.state_dim)

        whole_actor_actions_var = b_actor_actions_var.view(-1, self.seq_len, self.n_agents * self.action_dim)
        actor_actions_var = whole_actor_actions_var.view(-1, self.seq_len, self.n_agents,self.action_dim)

        whole_next_states_var = b_next_states_var.view(-1, self.seq_len, self.n_agents * self.state_dim)
        next_states_var = whole_next_states_var.view(-1, self.seq_len, self.n_agents, self.state_dim)
        nextactor_actions_list = []
        b_rewards_var = torch.transpose(b_rewards_var, 1,2)
        b_states_var = torch.transpose(b_states_var, 1,2)
        b_next_states_var = torch.transpose(b_next_states_var, 1,2)
        b_actor_actions_var = torch.transpose(b_actor_actions_var,1, 2)

        # for i in range(self.batch_size):
        #     # bool to binary
        #
        #     states_var = b_states_var[i].view(-1, self.n_agents, self.state_dim)
        #     actor_actions_var = b_actor_actions_var[i].view(-1, self.n_agents, self.action_dim)
        #     rewards_var = b_rewards_var[i].view(-1, self.n_agents, 1)
        #     next_states_var = b_next_states_var[i].view(-1, self.n_agents, self.state_dim)
        #     dones_var = b_dones_var[i].view(-1, 1)
        #
        #     nextactor_actions_var = torch.Tensor().to(self.device)
        nextactor_actions = []
        # Calculate next target actions for each agent
        for agent_id in range(self.n_agents):
            next_action_var = self.actors_target[agent_id](b_next_states_var[:, :, agent_id, :])
            if self.use_cuda:
                nextactor_actions.append(next_action_var)
            else:
                nextactor_actions.append(next_action_var)
        # Concatenate the next target actions into a single tensor
        nextactor_actions_torch = torch.cat(nextactor_actions, dim=1)
        nextactor_actions_list.append(nextactor_actions_torch)
        nextactor_actions_var = torch.stack(nextactor_actions_list, dim=0)
        nextactor_actions_var = nextactor_actions_var.view(-1, self.seq_len, b_actor_actions_var.size(2), b_actor_actions_var.size(3))
        whole_nextactor_actions_var = nextactor_actions_var.view(-1, self.seq_len, self.n_agents * self.action_dim)
        next_actor_actions_var = whole_nextactor_actions_var.view((-1, self.seq_len, self.n_agents, self.action_dim))
            # common critic
        agent_id = 0
        target_q = []
        current_q = []
        for b in range(self.batch_size):
            # target prediction
            # tar_perQ = self.critics_target[agent_id](whole_next_states_var[b], whole_nextactor_actions_var[b])
            tar_perQ = self.critics_target[agent_id](next_states_var[b], next_actor_actions_var[b])
            tar = self.reward_scale * b_rewards_var[b, :, agent_id, :] + self.reward_gamma * tar_perQ * (1. - b_dones_var[b])
            target_q.append(tar)
            curr_perQ = self.critics[agent_id](states_var[b], actor_actions_var[b])

            # curr_perQ = self.critics[agent_id](whole_states_var[b], whole_actor_actions_var[b])
            current_q.append(curr_perQ)
            errors[b] = errors[b] + F.mse_loss(curr_perQ, tar)
            idx = idxs[b]
            # print("errors",idx,errors)
            self.memory.update_priorities(idx, errors[b])
        # update critic network
        current_q = torch.stack(current_q, dim=0)
        target_q = torch.stack(target_q, dim=0)
        # c_loss = nn.MSELoss()(current_q, target_q)
        # c_loss.requires_grad_(True)
        # critic_loss_list.append(c_loss)
        critic_loss = nn.MSELoss()(current_q, target_q)
        # update target
        self.critics_optimizer[0].zero_grad()
        critic_loss.backward()
        loss.append(critic_loss)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critics[0].parameters(), self.max_grad_norm)
        self.critics_optimizer[0].step()
        self._soft_update_target(self.critics_target[0], self.critics[0])
        # different actors
        for agent_id in range(self.n_agents):

            newactor_actions_list = []
            # Calculate new actions for each agent
            # for b in range(self.batch_size):
            newactor_actions = []
            for agent_id in range(self.n_agents):
                newactor_action_var = self.actors[agent_id](b_states_var[:, :, agent_id, :])
                if self.use_cuda:
                    newactor_actions.append(
                        newactor_action_var)  # newactor_actions.append(newactor_action_var.data.cpu())
                else:
                    newactor_actions.append(newactor_action_var)  # newactor_actions.append(newactor_action_var.data)
            # Concatenate the new actions into a single tensor
            newactor_actions_torch = torch.cat(newactor_actions, dim=1)
            newactor_actions_list.append(newactor_actions_torch)
            newactor_actions_var = torch.stack(newactor_actions_list, dim=0)
            newactor_actions_var = newactor_actions_var.view(-1, self.seq_len, b_actor_actions_var.size(2), b_actor_actions_var.size(3))
            whole_newactor_actions_var = newactor_actions_var.view(-1, self.seq_len, self.n_agents * self.action_dim)
            actor_loss = []
            for b in range(self.batch_size):
                perQ = self.critics[0](states_var[b], newactor_actions_var[b])
                # perQ = self.critics[0](whole_states_var[b], whole_newactor_actions_var[b])
                actor_loss.append(perQ)
            actor_loss = torch.stack(actor_loss, dim=0)
            actor_loss = - actor_loss.mean()
            actor_loss.requires_grad_(True)
            self.actors_optimizer[agent_id].zero_grad()
            actor_loss.backward()
            loss.append(actor_loss)
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
            self.actors_optimizer[agent_id].step()
            self._soft_update_target(self.actors_target[agent_id], self.actors[agent_id])  # update target network

        loss = torch.mean(torch.stack(loss), dim=0)
        return loss

    def learn_ppo(self, batch_size=16, gae_lambda=0.95):
        # 从经验回放中采样

        idxs, states, actions, rewards, next_states,old_log_probs, old_values,dones = self.memory.sample_buffer(
            batch_size)
        old_log_probs = get_log_prob(actions)
        # 计算 GAE（广义优势估计）
        # ----------------------
        # Critic (Value) Update
        # ----------------------
        current_values = []
        target_values = []
        advantages = []
        dones = dones.view(batch_size, 1, 1 ).repeat(1, self.seq_len, 1)
        next_action_list = []
        ratio_all = []
        for agent_id in range(self.n_agents):
            # Get current policy distribution
            next_action = self.actors[agent_id](next_states)
            next_action_list.append(next_action)
            new_log_probs = get_log_prob(next_action)
            std = torch.exp(new_log_probs)
            dist = torch.distributions.Normal(next_action, std)
            # Importance sampling ratio
            ratio = torch.exp(new_log_probs[:, agent_id] - old_log_probs[:, agent_id])
            ratio_all.append(ratio)

            # Surrogate loss
            surr1 = ratio * advantages[agent_id]
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[agent_id]
            actor_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus
            entropy = dist.entropy().mean()
            entropy_loss = -self.entropy_coef * entropy


            policy_loss = actor_loss + entropy_loss
            self.actors_optimizer[agent_id].zero_grad()

            policy_loss.backward()
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(),self.max_grad_norm)

            self.actors_optimizer[agent_id].step()
            self._soft_update_target(self.actors_target[agent_id], self.actors[agent_id])
        # Centralized Value Function
        next_action_tensor = torch.stack(next_action_list)
        with torch.no_grad():
            next_values = self.critics_target[0](next_states.view(-1, self.n_agents * self.state_dim), next_action_tensor(-1, self.n_agents * self.state_dim))
            next_values = next_values.view(-1,  self.n_agents, self.seq_len,1)

        # Calculate TD targets and advantages
        for agent_id in range(self.n_agents):
            # Use centralized value function
            curr_value = self.critics(states.view(-1, self.n_agents * self.state_dim))
            curr_value = curr_value.view(-1, self.n_agents, self.seq_len, 1)[:, agent_id]

            # TD target with GAE
            delta = rewards[:, agent_id] + self.gamma * next_values[:, agent_id] * (1 - dones) - old_values[:, agent_id]
            advantage = self.compute_gae(delta, self.gae_lambda, self.gamma)
            advantages.append(advantage)

            target_value = advantage + old_values[:, agent_id]
            target_values.append(target_value)
            current_values.append(curr_value)

        # Value loss (MSE)
        value_loss = 0
        for agent_id in range(self.n_agents):
            value_loss += F.mse_loss(current_values[agent_id], target_values[agent_id])
        value_loss /= self.n_agents

        # ----------------------
        # Actor (Policy) Update
        # ----------------------
        actor_losses = []
        entropy_losses = []
        ratio_all = []
        policy_loss = []


        # Total policy loss

        # ----------------------
        # Optimization Step
        # ----------------------
        self.critics_optimizer.zero_grad()
        value_loss.backward()
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.critics.parameters(), self.max_grad_norm)
        self.critics_optimizer.step()


        # Update target networks
        self._soft_update_target(self.critics_target, self.critic)


        return value_loss

    def check_parameter_difference(self, model, loaded_state_dict):
        current_state_dict = model.state_dict()
        for name, param in current_state_dict.items():
            if name in loaded_state_dict:
                if not torch.equal(param, loaded_state_dict[name]):
                    # print(f"Parameter '{name}' has changed since the last checkpoint.")
                    return 1
                else:
                    # print(f"Parameter '{name}' has not changed since the last checkpoint.")
                    return 0
            else:
                print("Parameter '" + name + "' is not present in the loaded checkpoint.")
                exit()

    def getactionbound(self, a, b, x, i):
        x = (x - a) * (self.action_higher_bound[i] - self.action_lower_bound[i]) / (b - a) \
            + self.action_lower_bound[i]
        return x

    # choose an action based on state with random noise added for exploration in training
    def choose_actions(self, s):
        n_agent, seq_len, f = s.shape
        actor_action = torch.zeros((self.n_agents, seq_len, self.action_dim))

        for agent_id in range(self.n_agents):
            action_var = self.actors[agent_id](s[agent_id,:,:])
            actor_action[agent_id, :, :] = action_var
        val = torch.tensor([0.])
        return actor_action, val

    def get_flops(self):
        idxs, b_states_var, b_actor_actions_var, b_rewards_var, b_next_states_var, batch_log, batch_val, b_dones_var = self.memory.sample_buffer(
            self.batch_size)
        _,agent, _, _ = b_states_var.shape
        input_s = b_states_var[0, 0, :, :]
        input_a = b_actor_actions_var[0, : , :, :]
        # 开始计时
        torch.cuda.synchronize()  # 确保 GPU 空闲
        start_time = time.time()
        torch.cuda.synchronize()  # 等待所有 GPU 任务完成
        end_time = time.time()
        flops_a, params_a = profile(self.actors[0], inputs=(input_s,))
        # flops_c, params_c = profile(self.critics[0], inputs=(b_states_var[0, :, :, :],input_a, ))

        torch.cuda.synchronize()  # 等待所有 GPU 任务完成
        end_time = time.time()
        time_cost = (end_time - start_time) * 1000
        flop = (flops_a)
        return flop, time_cost, params_a


    def evaluateAtTraining(self, EVAL_EPISODES):
        # print(self.eval_episode_rewards)
        mean_reward = np.mean(np.array(self.Training_episode_rewards))
        self.Training_episode_rewards = []
        # self.mean_rewards.append(mean_reward)# to be plotted by the main function
        self.Training_episodes.append(self.n_episodes + 1)
        self.Training_results.append(mean_reward)
        arrayresults = np.array(self.Training_results)
        savetxt('./CSV/AtTraining/' + str(self.Benchmarks_mode) + str(self.InfdexofResult) + '.csv', arrayresults)
        # print("Episode:", self.n_episodes, "Episodic Reward:  Min mean Max : ", np.min(arrayresults), mean_reward, np.max(arrayresults))
    def save_model(self, path='marl_model.pth'):
        save_data = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'actor_optimizers': [actor_opt.state_dict() for actor_opt in self.actors_optimizer],
            'critic_optimizers': [critic_opt.state_dict() for critic_opt in self.critics_optimizer]
        }
        torch.save(save_data, path)
        print(f"save model to : {path}")

    def load_model(self,path='marl_model.pth'):
        checkpoint = torch.load(path)
        self.actors = [MambaActor(self.state_dim, self.e_dim, self.action_dim, self.actor_output_activation).to(self.device)] * self.n_agents

        self.critics = [CriticNetwork(self.critic_state_dim, self.critic_action_dim).to(self.device)] * 1

        for actor, state_dict in zip(self.actors, checkpoint['actors']):
            actor.load_state_dict(state_dict)
        for actor_opt, state_dict in zip(self.actors_optimizer, checkpoint['actor_optimizers']):
            actor_opt.load_state_dict(state_dict)
        for critic, state_dict in zip(self.critics, checkpoint['critics']):
            critic.load_state_dict(state_dict)
        for critic_opt, state_dict in zip(self.critics_optimizer, checkpoint['critic_optimizers']):
            critic_opt.load_state_dict(state_dict)
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)
        print(f"load model from {path} ")

    def release(self):
        del self.actors
        del self.critics

        gc.collect()


class MambaActor(nn.Module):
    """
    A network for actor with LSTM
    """

    def __init__(self, state_dim, e_dim, output_size, output_activation, hidden_size=256, init_w=3e-3):
        super(MambaActor, self).__init__()
        self.mamba1 = Mamba(d_model=hidden_size)
        # self.mamba2 = Mamba(d_model=hidden_size)

        self.e_dim = e_dim
        self.fc1 = nn.Linear(state_dim, hidden_size)
        # self.fc2 = nn.Linear(64, 32)
        # LSTM layer
        # self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, batch_first=True)
        # self.diffusion_fc = nn.Linear(hidden_size,hidden_size)
        self.diff = nn.Linear(hidden_size, hidden_size)
        # Output layer
        self.fc3 = nn.Linear(hidden_size, e_dim)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        self.diff_f1 =  nn.Parameter(torch.tensor(0.05))
        self.diff_f2 =  nn.Parameter(torch.tensor(0.05))

        # Activation function for the output
        self.output_activation = output_activation

    def forward(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(0)
        # state = torch.transpose(state, 0, 1)

        x = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.mamba1(x))

        # delta_x = F.relu(self.diff(x - torch.roll(x, shifts=1, dims=1)) ) # 计算时间差分
        # x1 = nn.functional.leaky_relu(x + self.mamba1(x + delta_x))
        # out = nn.functional.leaky_relu(x1 + self.mamba2(x1 + delta_x))
        # delta_x = self.diffusion_fc(x - torch.roll(x, shifts=1, dims=1))  # 计算时间差分

        # out = torch.cat((x[:,0:1,:], out), dim=1)
        # x = torch.cat((out[:,0:1,:],x1[:,:-1,:]), dim=1)
        # out = nn.functional.relu(self.mamba2(x1))

        # out_x = nn.functional.softmax(out, dim=-1)
        # diff_1 = x1[:,-1:,:] - out_x[:,1:,:]
        # diff_2 = x2[:,-1:,:] - out_x[:,1:,:]
        # diff = out_x[:,1:,:] + diff_2 - diff_1
        # out = nn.functional.softmax(x1 * self.diff_f1 - x2 * self.diff_f2, dim=-1)
        # out = nn.functional.sigmoid(torch.cat((x[:,0:1,:], diff), dim=1))
        # x1 = nn.functional.softmax(self.mamba1(x), dim=-1)
        # x2 = nn.functional.softmax(self.mamba2(x1), dim=-1)
        # delta_x1 = x[:,1:,:] - x1[:,:-1, :] * self.diff_f1
        # delta_x2 = (x2[:,-2, :]-x1[:,-1,:] * self.diff_f2).unsqueeze(1)
        # out = torch.cat([delta_x1, delta_x2], dim=1)

        if self.output_activation == nn.functional.softmax:
            out = self.output_activation(self.fc3(out), dim=-1)
        else:
            out = self.output_activation(self.fc3(out))
        # out = torch.transpose(out, 0, 1)
        # out = out.squeeze()

        probs = F.softmax(out)
        act = torch.argmax(probs, dim=-1).unsqueeze(-1)
        return act


class MambaCritic(nn.Module):
    """
    A network for critic
    """

    def __init__(self, state_dim, action_dim, output_size=1, init_w=3e-3, h_dim=512, device=torch.device("cuda:0")):
        super(MambaCritic, self).__init__()
        self.mamba1 = Mamba(d_model=h_dim)
        self.mamba2 = Mamba(d_model=h_dim)
        self.fc1 = nn.Linear(state_dim + action_dim,
                             h_dim)  # state_dim + action_dim = for the combined, equivalent of it for the per agent, and 1 for distinguisher
        # self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(h_dim, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        self.device = device
        self.dropout = nn.Dropout(0.1)


    def __call__(self, state, action):
        state = state.to(self.device)
        action = action.to(self.device)
        out = torch.cat([state, action], dim=-1)
        out = torch.transpose(out, 0, 1)
        out = self.fc1(out)
        if out.dim() == 2:
            out = out.unsqueeze(0)
        out = self.mamba1(out[:,:-1,:])
        # out = nn.functional.softmax(out, dim=-1)
        # diff = nn.functional.sigmoid(torch.cat((out[:,0:1,:], x), dim=1))
        x1 = nn.functional.softmax(self.mamba1(out), dim=-1)
        x2 = nn.functional.softmax(self.mamba2(x1), dim=-1)
        delta_x1 = out[:, 1:, :] - x1[:, :-1, :]
        delta_x2 = (x2[:, -2, :] - x1[:, -1, :]).unsqueeze(1)
        diff = torch.cat([delta_x1, delta_x2], dim=1)

        # out = nn.functional.tanh(out)
        out =  nn.functional.leaky_relu(self.fc3(diff))
        out = torch.transpose(out, 0, 1)
        out = out.flatten(1, 2)

        return out
class CriticNetwork(nn.Module):
    """
    A network for critic
    """

    def __init__(self, state_dim, action_dim, output_size=1, init_w=3e-3, device=torch.device("cuda:0")):
        super(CriticNetwork, self).__init__()
        # self.mamba1 = Mamba(d_model=state_dim + action_dim)
        # self.mamba2 = Mamba(d_model=state_dim + action_dim)
        self.mamba1 = Mamba(d_model=512)
        # self.mamba2 = Mamba(d_model=512)
        self.fc1 = nn.Linear(state_dim + action_dim,
                             512)  # state_dim + action_dim = for the combined, equivalent of it for the per agent, and 1 for distinguisher
        # self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(512, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        self.device = device

    def __call__(self, state, action):
        state = state.to(self.device)
        action = action.to(self.device)
        out = torch.cat([state, action], dim=-1)
        out = torch.transpose(out, 0, 1)
        if out.dim() == 2:
            out = out.unsqueeze(0)
        out = nn.functional.leaky_relu(self.fc1(out))
        out = nn.functional.leaky_relu(self.mamba1(out))
        out = nn.functional.leaky_relu(self.mamba2(out))

        # out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        out = torch.transpose(out, 0, 1)
        out = out.flatten(1, 2)

        return out


