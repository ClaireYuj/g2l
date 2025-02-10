import math
from torch.distributions import Categorical

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.util import init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class MultiFollower(nn.Module):
    # state: Task() + R(E) + priority(from leader)  + mob(10*2)
    # [batch, state, agent]
    def __init__(self, state_dim, action_dim, n_agent, seq_len, obs_dim=2, device=device,  hidden_dim=200, n_head=1, add_state_token=True):
        super(MultiFollower, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_agent = n_agent
        self.add_state_token = add_state_token
        self.obs_dim = obs_dim

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        self.state_encoder = nn.Sequential(nn.LayerNorm(self.state_dim),
                                           init_(nn.Linear(self.state_dim, hidden_dim), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(self.obs_dim),
                                         init_(nn.Linear(self.obs_dim, hidden_dim), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(hidden_dim)

        if self.add_state_token:
            self.class_token_encoding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.trunc_normal_(self.class_token_encoding, mean=0.0, std=0.02)
        self.device = device
        self.decoder = ActorCritic(n_head, n_agent, action_dim, 1, seq_len=seq_len,device=device, n_embed=self.hidden_dim)
        self.to(device)

    # def forward(self, state, obs, action, n_agent):
    #     # state = state.permute(1,0,2).to(device) #(seq, agent, state)
    #     # obs = obs.permute(1,0,2).to(device)
    #     state = state.to(device) #(seq, agent, state)
    #     n_agent, seq, c = state.shape
    #     obs = obs.to(device)
    #     action = action.to(device)
    #     obs_embeddings = self.obs_encoder(obs)
    #     state_embeddings = self.state_encoder(state[:, 0:1, :])
    #     obs_rep = torch.cat([state_embeddings, obs_embeddings], dim=1)
    #
    #     # one_hot_action = torch.zeros((self.seq_len, self.n_agent, self.action_dim)).to(self.device)
    #     # output_action = torch.zeros((self.seq_len, self.n_agent, 1), dtype=torch.long)
    #
    #     one_hot_action = F.one_hot(action.squeeze(-1), num_classes=self.action_dim).to(self.device)  # (batch, n_agent, action_dim)
    #     logit, v_loc = self.decoder(one_hot_action, obs_rep)
    #
    #
    #     distri = Categorical(logits=logit)
    #     action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    #     entropy = distri.entropy().unsqueeze(-1)
    #
    #     return v_loc, action_log, entropy

    def forward(self, state, obs, n_agent, deterministic=False):
    # def choose_actions(self, state, obs, n_agent, deterministic=False):
        # state = state.permute(1,0,2).to(device) #(seq, agent, state)
        # obs = obs.permute(1,0,2).to(device)
        state = state.to(device) #(seq, agent, state)
        obs = obs.to(device)
        obs_embeddings = self.obs_encoder(obs)
        state_embeddings = self.state_encoder(state[:, 0:1, :])
        obs_rep = torch.cat([state_embeddings, obs_embeddings], dim=1)
        # one_hot_action = torch.zeros((self.seq_len, self.n_agent, self.action_dim)).to(self.device)
        # output_action = torch.zeros((self.seq_len, self.n_agent, 1), dtype=torch.long)
        one_hot_action = torch.zeros((self.n_agent, self.seq_len, self.action_dim)).to(self.device)
        output_action = torch.zeros(( self.n_agent, self.seq_len, 1), dtype=torch.long)
        output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

        # for i in range(n_agent):
        for i in range(self.seq_len):
            logit, v_loc = self.decoder(one_hot_action, obs_rep)
            logit = logit[:, i, :]

            distri = Categorical(logits=logit)
            action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
            action_log = distri.log_prob(action)
            output_action[:, i, :] = action.unsqueeze(-1)
            output_action_log[:, i,:] = action_log.unsqueeze(-1)
            one_hot_action[:, i, :] = F.one_hot(action, num_classes=self.action_dim)


        # v_loc = v_loc.permute(1,0,2) #[seq_len, agent, state] -> [agent, seq_len, state]
        # output_action = output_action.permute(1,0,2)
        # output_action_log = output_action_log.permute(1,0,2)
        return v_loc, output_action, output_action_log

    def get_log_prob(self, actions):
        """
        计算给定动作的对数概率
        :param state: 当前状态
        :param action: 当前动作
        :return: 对数概率
        """
        dist = torch.distributions.Categorical(logits=actions)  # 离散分布
        action = dist.probs.argmax(dim=-1)
        action_log = dist.log_prob(action)
        return action_log

    def get_values(self, state, obs, actions, available_actions=None):
        # state unused

        state = state.to(self.device)
        obs = obs.to(self.device)
        actions = actions.to(self.device)

        # state_embeddings = self.state_encoder(state[:,0:1,:])
        obs_embeddings = self.obs_encoder(obs)
        # obs_embeddings = self.state_encoder(state)
        if self.add_state_token:
            obs_rep = torch.cat([self.class_token_encoding.expand(obs_embeddings.shape[0], -1, -1), obs_embeddings],
                                dim=1)  # batch_size, n_agent+1, embd
        else:
            state_embeddings = self.state_encoder(state[:, 0:1, :])
            obs_rep = torch.cat([state_embeddings, obs_embeddings], dim=1)  # batch_size, n_agent+1, embd

        one_hot_action = F.one_hot(actions.squeeze(-1), num_classes=self.action_dim).to(device)  # (batch, n_agent, action_dim)
        logit, v_loc = self.decoder(one_hot_action, obs_rep)
        if available_actions is not None:
            logit[available_actions == 0] = -1e10

        distri = Categorical(logits=logit)
        action_log = distri.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        entropy = distri.entropy().unsqueeze(-1)

        return v_loc






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
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # @==torch.matmul

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        # att = self.attn_drop(att)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs) ## @==torch.matmul
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

class ActorCritic(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, n_head, n_agent, action_dim, n_block, seq_len=10, device=device, n_embed=512):
        super(ActorCritic, self).__init__()

        self.device = device


        self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embed, bias=False), activate=True),
                                                    nn.GELU())
        self.seq_len = seq_len
        self.seq_ln = nn.LayerNorm(n_embed)
        self.obs_proj = nn.Sequential(
            init_(nn.Linear(n_embed, n_embed), activate=True), nn.GELU()
            # nn.LayerNorm(n_embed)
        )

        agent_dim = n_agent*seq_len

        self.obs_blocks = nn.Sequential(*[Block(n_embed, n_head, agent_dim+1, masked=False) for _ in range(n_block)])
        # self.ac_blocks = nn.Sequential(*[Block(n_embed, n_head, n_agent+1, masked=True) for _ in range(n_block)])
        self.ac_blocks = nn.Sequential(*[Block(n_embed, n_head, agent_dim+1, masked=True) for _ in range(n_block)])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(n_embed),
            init_(nn.Linear(n_embed, n_embed), activate=True), nn.GELU(),
            init_(nn.Linear(1*n_embed, n_embed))
        )
        self.action_head = nn.Sequential(init_(nn.Linear(n_embed, n_embed), activate=True), nn.GELU(), nn.LayerNorm(n_embed),
                                      init_(nn.Linear(n_embed, action_dim)))
        self.value_head = nn.Sequential(init_(nn.Linear(n_embed, n_embed), activate=True), nn.GELU(), nn.LayerNorm(n_embed),
                                  init_(nn.Linear(n_embed, 1)))

        # self.uncertain_embed = LearnableAbsolutePositionEmbedding(n_agent+1, n_embed) #uncertainty
        self.uncertain_embed = LearnableAbsolutePositionEmbedding(agent_dim+1, n_embed) #uncertainty



    def forward(self, action, obs_emb, state_dim=3):
        # obs_emb = torch.squeeze(obs_emb)#[seq,n_agent, state]
        # action = torch.squeeze(action)
        state_obs_rep = self.obs_blocks(obs_emb)

        state_rep = state_obs_rep[:,0:1,:]
        obs_rep = state_obs_rep[:,1:,:]
        action_rep = self.action_encoder(action.float())
        # action_rep = action_emb
        s_a = torch.cat([state_rep, action_rep], dim=1) # concat on n_agent_dim
        seq = self.uncertain_embed(s_a)
        # seq = torch.cat([state_rep, action_rep], dim=1) # v3
        x = self.ac_blocks(seq)

        x_cut = x[:,:-1,:]
        x_cut = x_cut + obs_rep
        x = torch.cat([x[:,-1:,:].clone(), x_cut], dim=1)
        # x[:,:-1,:] += obs_rep # in-place
        x = x + self.out_proj(x) # residual
        x = self.seq_ln(x)
        logit = self.action_head(x)[:, :-1, :]
        v_loc = self.value_head(x)[:, :-1, :]

        return logit, v_loc

class LearnableAbsolutePositionEmbedding(nn.Module):
    ## learn the mobility
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
            x = x + self.embeddings(position_ids)[None, :, :]
            return x