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


class MAPPO(object):
    def __init__(self,n_agents, state_dim, action_dim, seq_len, action_lower_bound,e_dim,
                 action_higher_bound,memory_capacity=10000,
                 InfdexofResult=0,
                 target_tau=1, reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_output_activation=torch.tanh, actor_lr=0.0001, critic_lr=0.0001,
                 optimizer_type="adam", max_grad_norm=True, batch_size=1, episodes_before_train=64,
                 epsilon_start=1, epsilon_end=0.01, epsilon_decay=None, use_cuda=True, Benchmarks_mode=None):
        super(MAPPO, self).__init__()
        self.n_agents = n_agents

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lower_bound = action_lower_bound
        self.action_higher_bound = action_higher_bound
        self.n_episodes = 0
        self.seq_len = seq_len
        self.e_dim = e_dim
        self.memory = PrioritizedReplayBuffer(memory_capacity,self.state_dim, action_dim=self.e_dim, n_agent=self.n_agents, seq_len = self.seq_len, mode='maddpg')
        self.actor_output_activation = actor_output_activation
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.max_grad_norm = max_grad_norm
        self.batch_size = 1
        # params for epsilon greedy
        self.device = torch.device("cuda:0")
        self.reward_scale = 1.
        self.reward_gamma = 0.95

        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.target_tau = target_tau
        self.actors = [ActorNetwork(self.state_dim, e_dim, self.action_dim, self.actor_output_activation).to(self.device)] * self.n_agents

        critic_state_dim = self.n_agents * self.state_dim
        critic_action_dim = self.n_agents * self.action_dim
        self.critic = CriticNetwork(critic_state_dim, self.n_agents).to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.actors_target = deepcopy(self.actors)
        # to ensure target network and learning network has the same weights

        if optimizer_type == "adam":
            self.actors_optimizer = [Adam(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        elif optimizer_type == "rmsprop":
            self.actors_optimizer = [RMSprop(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)

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
        self.gamma = 0.9
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01


    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)


    # train on a sample batch
    def train(self, batch_size=8, gae_lambda=0.95):
        # 从经验回放中采样

        idxs, states, actions, rewards, next_states, log_probs, q_values,dones = self.memory.sample_buffer(
            batch_size)
        # old_values = q_values.clone().detach()
        old_values = q_values.clone().detach()

        # 计算 GAE（广义优势估计）
        # ----------------------
        # Critic (Value) Update
        # ----------------------
        current_values = []
        target_values = []
        advantages = []
        dones = dones.view(batch_size,1, 1, 1 ).repeat(1, self.n_agents, self.seq_len, 1)
        next_values = self.critic_target(next_states.view(-1, self.n_agents * self.state_dim))
        next_values = next_values.view(-1, self.n_agents, self.seq_len, 1)
        for b in range(self.batch_size):
        # Centralized Value Function
        #     with torch.no_grad():

            # Calculate TD targets and advantages
            # for agent_id in range(self.n_agents):
                # Use centralized value function
            curr_value = self.critic(states[b].view(-1, self.n_agents * self.state_dim))
            current_values = curr_value.view( self.n_agents, self.seq_len, 1)
            # curr_value = curr_value.view(-1, self.n_agents, self.seq_len, 1)[:, agent_id]
            delta = rewards[b] + self.gamma * next_values[b] * (1 - dones[b]) - old_values[b]
            with torch.no_grad():
                old_log_probs = self.get_log_prob(actions[b])

                advantages = self.compute_gae(delta, self.gae_lambda, self.gamma)
            # advantages.append(advantage)
            target_values = advantages + old_values[b]
            # target_values.append(target_value)
            # current_values.append(curr_value)
            ratio_all = []
            policy_losses = []
            value_loss = 0
            value_loss = value_loss + F.mse_loss(current_values, target_values)

            value_loss = value_loss / self.n_agents

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            for agent_id in range(self.n_agents):
                # Get current policy distribution
                next_action, _ = self.actors[agent_id](next_states[b, agent_id])

                new_log_probs = self.get_log_prob(next_action)
                std = torch.exp(new_log_probs)
                dist = torch.distributions.Normal(next_action, std)
                # Importance sampling ratio
                ratio = torch.exp(new_log_probs - old_log_probs) + 1e-8
                ratio_all.append(ratio)

                # Surrogate loss
                surr1 = ratio * advantages[agent_id]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[agent_id]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                entropy = dist.entropy().mean()
                entropy_loss = -self.entropy_coef * entropy

                policy_loss = actor_loss + entropy_loss
                # policy_losses.append(actor_loss)
                self.actors_optimizer[agent_id].zero_grad()

                policy_loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)

                self.actors_optimizer[agent_id].step()
                self._soft_update_target(self.actors_target[agent_id], self.actors[agent_id])

            # for agent_id in range(self.n_agents):
            #     self.actors_optimizer[agent_id].zero_grad()
            #
            #     policy_losses[agent_id].backward()
            #     if self.max_grad_norm:
            #         torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
            #
            #     self.actors_optimizer[agent_id].step()
            #     self._soft_update_target(self.actors_target[agent_id], self.actors[agent_id])
            self._soft_update_target(self.critic_target, self.critic)

        self.memory.update_priorities(idxs[b], value_loss)

        # Total policy loss
        return value_loss

    def compute_gae(self, deltas, gamma=0.99, lambda_=0.95):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        advantage = 0
        for delta in reversed(deltas):
            advantage = delta + gamma * lambda_ * advantage
            advantages.insert(0, advantage)
        adv = torch.stack(advantages)
        return adv

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

    # # choose an action based on state with random noise added for exploration in training
    # def choose_actions(self, s):
    #     n_agent, seq_len, f = s.shape
    #     actor_action = torch.zeros((self.n_agents, seq_len, self.actors))
    #     for agent_id in range(self.n_agents):
    #         action_var = self.actors[agent_id](s[agent_id,:,:])
    #         actor_action[agent_id, :, :] = action_var
    #     val = torch.tensor([0.])
    #     return actor_action,val
    def choose_actions(self, s):
        n_agent, seq_len, f = s.shape
        actor_action = torch.zeros((self.n_agents, seq_len, self.e_dim))
        edges =  torch.zeros((self.n_agents, seq_len, self.action_dim))
        for agent_id in range(self.n_agents):
            action_var, edge = self.actors[agent_id](s[agent_id,:,:])
            actor_action[agent_id, :, :] = action_var
            edges[agent_id, :, :] = edge
        val = torch.tensor([0.])
        return actor_action, edges, val

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
            'critic': [critic.state_dict() for critic in self.critic],
            'actor_optimizers': [actor_opt.state_dict() for actor_opt in self.actors_optimizer],
            'critic_optimizers': [critic_opt.state_dict() for critic_opt in self.critic_optimizer]
        }
        torch.save(save_data, path)
        print(f"save model to : {path}")

    def load_model(self,path='marl_model.pth'):
        checkpoint = torch.load(path)
        for actor, state_dict in zip(self.actors, checkpoint['actors']):
            actor.load_state_dict(state_dict)
        for actor_opt, state_dict in zip(self.actors_optimizer, checkpoint['actor_optimizers']):
            actor_opt.load_state_dict(state_dict)
        for critic, state_dict in zip(self.critic, checkpoint['critic']):
            critic.load_state_dict(state_dict)
        for critic_opt, state_dict in zip(self.critic_optimizer, checkpoint['critic_optimizers']):
            critic_opt.load_state_dict(state_dict)
        print(f"load model from {path} ")

    def get_log_prob(self, actions):
        """
        计算给定动作的对数概率
        :param state: 当前状态
        :param action: 当前动作
        :return: 对数概率
        """
        logits = torch.where(torch.isnan(actions), torch.zeros_like(actions), actions)
        logits = torch.clamp(logits, min=-1e2, max=1e2)  # 限制 logits 的范围
        dist = torch.distributions.Categorical(logits=logits)  # 离散分布
        action = dist.probs.argmax(dim=-1)
        action_log = dist.log_prob(action).unsqueeze(-1)
        return action_log
class ActorNetwork(nn.Module):
    """
    A network for actor
    """




    def __init__(self, state_dim, e_dim, output_size, output_activation, init_w=3e-3):
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
        action_mean, act = self.forward(x)
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

