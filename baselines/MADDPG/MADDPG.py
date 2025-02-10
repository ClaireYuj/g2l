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


class CCMADDPG(object):
    def __init__(self,n_agents, state_dim, action_dim, action_lower_bound,
                 action_higher_bound,memory_capacity=10000,
                 InfdexofResult=0,
                 target_tau=1, reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_output_activation=torch.tanh, actor_lr=0.0001, critic_lr=0.001,
                 optimizer_type="adam", max_grad_norm=None, batch_size=64, episodes_before_train=64,
                 epsilon_start=1, epsilon_end=0.01, epsilon_decay=None, use_cuda=False, Benchmarks_mode=None):
        super(CCMADDPG, self).__init__()

        self.Benchmarks_mode = Benchmarks_mode
        print(Benchmarks_mode)
        self.n_agents = n_agents

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lower_bound = action_lower_bound
        self.action_higher_bound = action_higher_bound
        self.env_state = self.env_Benchmark.reset_mec()
        self.n_episodes = 0

        self.memory = PrioritizedReplayBuffer(memory_capacity)
        self.actor_output_activation = actor_output_activation
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train
        # params for epsilon greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        if epsilon_decay == None:
            print("epsilon_decay is NOne")
            exit()
        else:
            self.epsilon_decay = epsilon_decay
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.target_tau = target_tau
        self.actors = [ActorNetwork(self.state_dim, self.action_dim, self.actor_output_activation)] * self.n_agents
        critic_state_dim = self.n_agents * self.state_dim
        critic_action_dim = self.n_agents * self.action_dim
        self.critics = [CriticNetwork(critic_state_dim, critic_action_dim)] * 1
        # to ensure target network and learning network has the same weights
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)
        if optimizer_type == "adam":
            self.actors_optimizer = [Adam(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critics_optimizer = [Adam(c.parameters(), lr=self.critic_lr) for c in self.critics]
        elif optimizer_type == "rmsprop":
            self.actors_optimizer = [RMSprop(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critics_optimizer = [RMSprop(c.parameters(), lr=self.critic_lr) for c in self.critics]
        if self.use_cuda:
            for i in range(self.n_agents):
                self.actors[i].cuda()
                self.critics[i].cuda()
                self.actors_target[i].cuda()
                self.critics_target[i].cuda()
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
        if self.n_episodes <= self.episodes_before_train:
            return
        tryfetch = 0
        idxs, states_var, actor_actions_var, rewards_var, next_states_var, batch_log, batch_val, dones_var = self.memory.sample_buffer(
            self.batch_size)
        while tryfetch < 3:

            # print("idxs, is_weights", len(idxs), len(is_weights))
            mini_batch = np.array(mini_batch, dtype=object).transpose()
            if any(not isinstance(arr, np.ndarray) for arr in mini_batch[0]) or any(
                    not isinstance(arr, np.ndarray) for arr in mini_batch[2]):
                if tryfetch < 3:
                    tryfetch += 1
                else:
                    print("mini_batch = ", mini_batch)
                    exit()
            else:
                break
        errors = np.zeros(self.batch_size)
        states = np.vstack(mini_batch[0])
        actor_actions = np.vstack(mini_batch[1])
        rewards = np.vstack(mini_batch[2])
        next_states = np.vstack(mini_batch[3])
        dones = mini_batch[4]

        # bool to binary
        dones = dones.astype(int)
        states_var = states_var.view(-1, self.n_agents, self.state_dim)
        actor_actions_var = actor_actions_var.view(-1, self.n_agents, self.action_dim)
        rewards_var = rewards_var.view(-1, self.n_agents, 1)
        next_states_var = next_states_var.view(-1, self.n_agents, self.state_dim)
        dones_var = dones_var.view(-1, 1)
        whole_states_var = states_var.view(-1, self.n_agents * self.state_dim)
        whole_actor_actions_var = actor_actions_var.view(-1, self.n_agents * self.action_dim)
        whole_next_states_var = next_states_var.view(-1, self.n_agents * self.state_dim)

        nextactor_actions = []
        # Calculate next target actions for each agent
        for agent_id in range(self.n_agents):
            next_action_var = self.actors_target[agent_id](next_states_var[:, agent_id, :])
            if self.use_cuda:
                nextactor_actions.append(next_action_var)
            else:
                nextactor_actions.append(next_action_var)
        # Concatenate the next target actions into a single tensor
        nextactor_actions_var = torch.cat(nextactor_actions, dim=1)
        nextactor_actions_var = nextactor_actions_var.view(-1, actor_actions_var.size(1), actor_actions_var.size(2))
        whole_nextactor_actions_var = nextactor_actions_var.view(-1, self.n_agents * self.action_dim)

        # common critic
        agent_id = 0
        target_q = []
        current_q = []
        for b in range(self.batch_size):
            # target prediction
            tar_perQ = self.critics_target[agent_id](whole_next_states_var[b], whole_nextactor_actions_var[b])
            tar = self.reward_scale * rewards_var[b, agent_id, :] + self.reward_gamma * tar_perQ * (1. - dones_var[b])
            target_q.append(tar)
            curr_perQ = self.critics[agent_id](whole_states_var[b], whole_actor_actions_var[b])
            current_q.append(curr_perQ)
            errors[b] += (curr_perQ - tar) ** 2
        # update critic network
        current_q = torch.stack(current_q, dim=0)
        target_q = torch.stack(target_q, dim=0)
        critic_loss = nn.MSELoss()(current_q, target_q)
        critic_loss.requires_grad_(True)
        self.critics_optimizer[agent_id].zero_grad()
        critic_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), self.max_grad_norm)
        self.critics_optimizer[agent_id].step()
        self._soft_update_target(self.critics_target[agent_id], self.critics[agent_id])  # update target

        # different actors
        for agent_id in range(self.n_agents):
            newactor_actions = []
            # Calculate new actions for each agent
            for agent_id in range(self.n_agents):
                newactor_action_var = self.actors[agent_id](states_var[:, agent_id, :])
                if self.use_cuda:
                    newactor_actions.append(
                        newactor_action_var)  # newactor_actions.append(newactor_action_var.data.cpu())
                else:
                    newactor_actions.append(newactor_action_var)  # newactor_actions.append(newactor_action_var.data)
            # Concatenate the new actions into a single tensor
            newactor_actions_var = torch.cat(newactor_actions, dim=1)
            newactor_actions_var = newactor_actions_var.view(-1, actor_actions_var.size(1), actor_actions_var.size(2))
            whole_newactor_actions_var = newactor_actions_var.view(-1, self.n_agents * self.action_dim)
            actor_loss = []
            for b in range(self.batch_size):
                perQ = self.critics[0](whole_states_var[b], whole_newactor_actions_var[b])
                actor_loss.append(perQ)
            actor_loss = torch.stack(actor_loss, dim=0)
            actor_loss = - actor_loss.mean()
            actor_loss.requires_grad_(True)
            self.actors_optimizer[agent_id].zero_grad()
            actor_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
            self.actors_optimizer[agent_id].step()
            self._soft_update_target(self.actors_target[agent_id], self.actors[agent_id])  # update target network
        for i in range(self.batch_size):
            idx = idxs[i]
            # print("errors",idx,errors)
            self.memory.update(idx, errors[i])


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
    def choose_action(self, s):
        actor_action = torch.zeros((self.n_agents, self.action_dim))
        for agent_id in range(self.n_agents):
            action_var = self.actors[agent_id](s[agent_id,:,:])
            actor_action[agent_id] = action_var

        return actor_action

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


class ActorNetwork(nn.Module):
    """
    A network for actor
    """

    def __init__(self, state_dim, output_size, output_activation, init_w=3e-3):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        # activation function for the output
        self.output_activation = output_activation

    def __call__(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        if self.output_activation == nn.functional.softmax:
            out = self.output_activation(self.fc3(out), dim=-1)
        else:
            out = self.output_activation(self.fc3(out))
        return out


class CriticNetwork(nn.Module):
    """
    A network for critic
    """

    def __init__(self, state_dim, action_dim, output_size=1, init_w=3e-3):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim,
                             512)  # state_dim + action_dim = for the combined, equivalent of it for the per agent, and 1 for distinguisher
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state, action):
        out = torch.cat([state, action], 0)
        out = nn.functional.relu(self.fc1(out))
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out

