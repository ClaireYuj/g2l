import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from utils.util import init, normalization
from torch.distributions import Categorical, Normal



def discrete_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False):
    one_hot_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        logit, v_loc = decoder(one_hot_action, obs_rep)
        logit = logit[:, i, :]
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10

        distri = Categorical(logits=logit)
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)
        one_hot_action[: , i, :] = F.one_hot(action, num_classes=action_dim)
    return v_loc, output_action, output_action_log


def discrete_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None):
    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim).to(**tpdv)  # (batch, n_agent, action_dim)
    logit, v_loc = decoder(one_hot_action, obs_rep)
    if available_actions is not None:
        logit[available_actions == 0] = -1e10

    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    return v_loc, action_log, entropy


def continuous_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                  deterministic=False):
    
    output_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32).to(**tpdv)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        act_mean, v_loc = decoder(output_action, obs_rep)
        act_mean = act_mean[:, i, :]
        action_std = torch.sigmoid(decoder.log_std) * 0.5

        # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
        # distri = Normal(act_mean, log_std.exp())
        distri = Normal(act_mean, action_std)
        action = act_mean if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action
        output_action_log[:, i, :] = action_log

        # print("act_mean: ", act_mean)
        # print("action: ", action)

    return v_loc, output_action, output_action_log


def continuous_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv):

    act_mean, v_loc = decoder(action, obs_rep)
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    distri = Normal(act_mean, action_std)

    # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
    # distri = Normal(act_mean, log_std.exp())

    action_log = distri.log_prob(action)    #计算value在定义的正态分布中对应的概率的对数,计算当前策略下的动作分布,然后计算旧策略采样下的动作在本分布下的概率对数
    entropy = distri.entropy()
    return v_loc, action_log, entropy

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, seq_size, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        attn_pdrop = 0.1
        resid_pdrop = 0.1
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(seq_size + 1, seq_size + 1))
                             .view(1, 1, seq_size + 1, seq_size + 1))

        self.att_bp = None

    def forward(self, x):
        B, L, D = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs) L:num_agents
        q = self.query(x).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(x).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        # att = self.attn_drop(att)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        # y = self.resid_drop(self.proj(y))
        y = self.proj(y)
        return y


class Block(nn.Module):
    def __init__(self, n_embed, n_head, seq_size, masked):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.attn = SelfAttention(n_embed, n_head, seq_size, masked)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embed, 1*n_embed), activate=True),
            nn.GELU(),
            init_(nn.Linear(1*n_embed, n_embed))
            # nn.Dropout(resid_pdrop)
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x
    
class LearnableAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self.is_absolute = True
        self.embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.register_buffer('position_ids', torch.arange(max_position_embeddings))

    def forward(self, x):
        """
        return (b l d) / (b h l d)
        """
        position_ids = self.position_ids[:x.size(-2)]

        if x.dim() == 3:
            return x + self.embeddings(position_ids)[None, :, :]

class ActorCritic(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, n_embed, n_head, n_agent, action_dim, n_block, action_type='Discrete', device="CPU"):
        super(ActorCritic, self).__init__()

        self.device = device
        self.n_agent = n_agent
        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

        if action_type == 'Discrete':
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embed, bias=False), activate=True),
                                                    nn.GELU())
        else:
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embed), activate=True), nn.GELU())

        self.seq_ln = nn.LayerNorm(n_embed)
        self.obs_proj = nn.Sequential(
            init_(nn.Linear(n_embed, n_embed), activate=True), nn.GELU()
            # nn.LayerNorm(n_embed)
        )

        self.obs_blocks = nn.Sequential(*[Block(n_embed, n_head, n_agent+1, masked=False) for _ in range(n_block)])
        # self.ac_blocks = nn.Sequential(*[Block(n_embed, n_head, n_agent+1, masked=True) for _ in range(n_block)])
        self.ac_blocks = nn.Sequential(*[Block(n_embed, n_head, n_agent+1, masked=True) for _ in range(n_block)])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(n_embed),
            init_(nn.Linear(n_embed, n_embed), activate=True), nn.GELU(),
            init_(nn.Linear(1*n_embed, n_embed))
        )
        self.action_head = nn.Sequential(init_(nn.Linear(n_embed, n_embed), activate=True), nn.GELU(), nn.LayerNorm(n_embed),
                                      init_(nn.Linear(n_embed, action_dim)))
        self.value_head = nn.Sequential(init_(nn.Linear(n_embed, n_embed), activate=True), nn.GELU(), nn.LayerNorm(n_embed),
                                  init_(nn.Linear(n_embed, 1)))
        self.pos_embed = LearnableAbsolutePositionEmbedding(1+n_agent, n_embed)

    def zero_std(self):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(self.device)
            self.log_std.data = log_std


    def forward(self, action, obs_emb):
        state_obs_rep = self.obs_blocks(obs_emb)
        state_rep = state_obs_rep[:,0:1,:]
        obs_rep = state_obs_rep[:,1:,:]
        action_emb = self.action_encoder(action)
        action_rep = action_emb

        seq = self.pos_embed(torch.cat([state_rep, action_rep], dim=1))
        # seq = torch.cat([state_rep, action_rep], dim=1) # v3
        x = self.ac_blocks(seq)
        x[:,:-1,:] += obs_rep
        x = x + self.out_proj(x)
        x = self.seq_ln(x)
        logit = self.action_head(x)[:, :-1, :]
        v_loc = self.value_head(x)[:, :-1, :]
        return logit, v_loc

class Steer(nn.Module):

    def __init__(self, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False, add_state_token=True):
        super(Steer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        self.add_state_token = add_state_token


        self.state_dim = state_dim
        self.encode_state = encode_state
        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
        
        self.ln = nn.LayerNorm(n_embd)
        
        if self.add_state_token:
            self.class_token_encoding = nn.Parameter(torch.zeros(1, 1, n_embd))
            nn.init.trunc_normal_(self.class_token_encoding, mean=0.0, std=0.02)
        self.decoder = ActorCritic(n_embd, n_head, n_agent, action_dim, n_block, action_type, self.device)
        self.device = device
        self.to(device)

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    def forward(self, state, obs, n_agent, deterministic=False):

        state = state.permute(1,0,2).to(self.device)
        b, n, s = state.shape
        obs = obs.permute(1,0,2).to(self.device)
        obs_embeddings = self.obs_encoder(obs)
        state_embeddings = self.state_encoder(state[:, 0:1, :])
        obs_rep = torch.cat([state_embeddings, obs_embeddings], dim=1)
        one_hot_action = torch.zeros((b, n, self.action_dim)).to(self.device)
        output_action = torch.zeros((b, n, 1), dtype=torch.long)
        output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

        for i in range(n_agent):
            logit, v_loc = self.decoder(one_hot_action, obs_rep)
            # logit = logit[:, i, :]
            logit = logit[i,:,:]


            distri = Categorical(logits=logit)
            action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
            action_log = distri.log_prob(action)
            output_action[ i, :, :] = action.unsqueeze(-1)
            output_action_log[i, :,:] = action_log.unsqueeze(-1)
            one_hot_action[i, :, :] = F.one_hot(action, num_classes=self.action_dim)

            # output_action[:, i, :] = action.unsqueeze(-1)
            # output_action_log[:, i, :] = action_log.unsqueeze(-1)
            # one_hot_action[:, i, :] = F.one_hot(action, num_classes=self.action_dim)
        v_loc = v_loc.permute(1,0,2) #[seq_len, agent, state] -> [agent, seq_len, state]
        output_action = output_action.permute(1,0,2)
        output_action_log = output_action_log.permute(1,0,2)
        return v_loc, output_action, output_action_log

    def choose_actions(self, state, obs, n_agent, deterministic=False):
        # state = state.permute(1,0,2).to(device) #(seq, agent, state)
        # obs = obs.permute(1,0,2).to(device)
        state = state.to(self.device) #(seq, agent, state)
        obs = obs.to(self.device)
        obs_embeddings = self.obs_encoder(obs)
        state_embeddings = self.state_encoder(state[:, 0:1, :])
        obs_rep = torch.cat([state_embeddings, obs_embeddings], dim=1)
        # one_hot_action = torch.zeros((self.seq_len, self.n_agent, self.action_dim)).to(self.device)
        # output_action = torch.zeros((self.seq_len, self.n_agent, 1), dtype=torch.long)
        one_hot_action = torch.zeros((self.n_agent, self.seq_len, self.action_dim)).to(self.device)
        output_action = torch.zeros(( self.n_agent, self.seq_len, 1), dtype=torch.long)
        output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

        # for i in range(n_agent):
        for i in range(self.n_agent):
            logit, v_loc = self.decoder(one_hot_action, obs_rep)
            logit = logit[:, i, :]

            distri = Categorical(logits=logit)
            action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
            action_log = distri.log_prob(action)
            output_action[ i, :, :] = action.unsqueeze(-1)
            output_action_log[i, :,:] = action_log.unsqueeze(-1)
            one_hot_action[i, :, :] = F.one_hot(action, num_classes=self.action_dim)


        # v_loc = v_loc.permute(1,0,2) #[seq_len, agent, state] -> [agent, seq_len, state]
        # output_action = output_action.permute(1,0,2)
        # output_action_log = output_action_log.permute(1,0,2)
        return v_loc, output_action, output_action_log

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, actions, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param actions: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim) # time_steps*n_rollout_threads*num_agents/batch_size, dim -> -1, num_agents, dim
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        actions = actions.reshape(-1, self.num_agents, self.act_num)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        action_log_probs, values, entropy = self.transformer(cent_obs, obs, actions, available_actions)

        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.contiguous().view(-1, 1)
        entropy = entropy.view(-1, self.act_num)

        if self._use_policy_active_masks and active_masks is not None:
            entropy = (entropy*active_masks).sum()/active_masks.sum()
        else:
            entropy = entropy.mean()

        return values, action_log_probs, entropy



    def learn(self, memory, batch_size=2, gamma=0.99, gae_lambda = 0.95):
        # indices = np.random.choice(self.memory_capacity, size=BATCH_SIZE)

        # indices = np.random.choice(self.memory_capacity, size=BATCH_SIZE) # generate data to train
        # bt = self.memory[indices, :]
        bs, ba, br, bs_, bter = memory.sample_buffer(batch_size)
        # Convert to PyTorch tensors
        bs = torch.from_numpy(normalization(bs)).clone().detach().to(torch.float32).to(self.device)
        ba = torch.from_numpy(normalization(ba)).clone().detach().to(torch.float32).to(self.device)
        br = torch.from_numpy(normalization(br)).clone().detach().to(torch.float32).to(self.device)
        bs_ = torch.from_numpy(normalization(bs_)).clone().detach().to(torch.float32).to(self.device)
        bs = bs.flatten(1, 2) #[batch, n_agent, seq, action] -> [batch,n_agent*seq, action]
        ba = ba.flatten(1, 2)
        bs_ = bs_.flatten(1, 2)
        # Critic network update
        q_values = self.critic(bs, ba)
        # print("update networks")
        with torch.no_grad():
            next_actions = self.actor_target(bs_)
            next_q_values = self.critic_target(bs_, next_actions)
            target_q_values = br + gamma * next_q_values #TD
            advantage = torch.zeros_like(next_q_values, dtype=torch.float32)
            gae = 0
            for step in reversed(range(br.shape[0])):
                delta = br[step]  + gamma * q_values[step] - next_q_values[step]
                gae = delta + gamma * gae_lambda * gae  # gae_lambda=0.9 使用lambda平滑GAE
                advantage[step] = gae

        critic_loss = torch.mean((q_values - target_q_values) ** 2) #原本为正
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_max_norm)

        self.critic_optimizer.step()
        # Actor network update
        actions_pred = self.actor(bs)

        # Compute the ratio (new policy over old policy)
        log_prob_new = self.actor.get_log_prob(actions_pred)  # 获取当前动作的log概率
        log_prob_target = self.actor_target.get_log_prob(next_actions)  # 获取旧策略的log概率

        ratio = torch.exp(log_prob_new - log_prob_target)  # 比率 r_t(θ)
        # Compute the clipped loss for PPO
        epsilon = 0.2
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        advantage = advantage.squeeze()
        actor_loss = -torch.mean(torch.min(ratio * advantage, clipped_ratio * advantage))  # 裁剪的目标函数
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_max_norm)

        self.actor_optimizer.step()

        # Perform soft update of target networks
        self.soft_update(self.actor, self.actor_target, self.tau)
        self.soft_update(self.critic, self.critic_target, self.tau)
        return actor_loss, critic_loss
