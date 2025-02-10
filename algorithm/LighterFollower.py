import math
from torch.distributions import Categorical, Normal

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.util import init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class LighterFollower(nn.Module):
    # state: Task() + R(E) + priority(from leader)  + mob(10*2)
    # [batch, state, agent]
    def __init__(self, state_dim, action_dim, n_agent, seq_len, candidate_choice=1,obs_dim=2, device=device,  hidden_dim=256, n_head=1, add_state_token=True):
        super(LighterFollower, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_agent = n_agent
        self.add_state_token = add_state_token
        self.obs_dim = obs_dim
        self.candidate_choice = candidate_choice

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        self.state_encoder = nn.Sequential(nn.LayerNorm(self.state_dim),
                                           init_(nn.Linear(self.state_dim, hidden_dim), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(self.obs_dim),
                                         init_(nn.Linear(self.obs_dim, hidden_dim), activate=True), nn.GELU())

        # self.ln = nn.LayerNorm(hidden_dim)

        self.device = device
        self.decoder = ActorCritic(n_head, n_agent, 1, 1, seq_len=seq_len,device=device, n_embed=self.hidden_dim)

        # self.decoder = ActorCritic(n_head, n_agent, self.candidate_choice+1, 1, seq_len=seq_len,device=device, n_embed=self.hidden_dim)
        self.to(device)



    # def forward(self, state, obs, action):
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
        """
        Discrete
        """
        # state = state.permute(1,0,2).to(device) #(seq, agent, state)
        # obs = obs.permute(1,0,2).to(device)
        # candidate_edge = state[:, :, -(self.candidate_choice):].to(device) # action_dim = n_candidate_num+1->n_candidate_num = action-1
        if state.dim() == 2:
            state = state.unsqueeze(0).clone()
            obs = obs.unsqueeze(0).clone()
        candidate_edge = state[:, :, -(self.candidate_choice):].to(device) # action_dim = n_candidate_num+1->n_candidate_num = action-1

        state = state.to(device) #(seq, agent, state)
        obs = obs.to(device)
        obs_embeddings = self.obs_encoder(obs)
        state_embeddings = self.state_encoder(state[:, 0:1, :])
        obs_rep = torch.cat([state_embeddings, obs_embeddings], dim=1)
        n_agent, seq_len, _ = state.shape

        # one_hot_action = torch.zeros((n_agent, seq_len, self.candidate_choice+1)).to(self.device)
        one_hot_action = torch.zeros((n_agent, seq_len, 1)).to(self.device)

        # output_action = torch.zeros(( self.n_agent, self.seq_len, 1), dtype=torch.long)
        output_action = torch.zeros(( n_agent, seq_len, 1), dtype=torch.long)

        output_action_log = torch.zeros_like(output_action, dtype=torch.float32)
        logit, v_loc = self.decoder(one_hot_action, obs_rep)

        for i in range(n_agent):
            for j in range(self.seq_len):
                # logit = logit[i, :, :]
                out = logit[i, j, :]
                # out = out[i, :]
                distri = Categorical(logits=out)
                action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
                action_log = distri.log_prob(action)
                if action.item() >= self.candidate_choice:
                    choice = torch.tensor([-1])
                else:
                    choice = candidate_edge[i, -1, action]

                output_action[i, j, :] = choice.unsqueeze(-1)
                output_action_log[i, j,:] = action_log.unsqueeze(-1)

        return output_action
        # return output_action, v_loc, output_action_log

    def get_log_prob(self, actions):
        """
        get the log proabilities
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
        log_std = torch.ones(action_dim)
        # log_std = torch.zeros(action_dim)
        self.log_std = torch.nn.Parameter(log_std)
        self.relu = nn.LeakyReLU()


    def forward(self, action, obs_emb, state_dim=3):
        # obs_emb = torch.squeeze(obs_emb)#[seq,n_agent, state]
        # action = torch.squeeze(action)
        # state_obs_rep = self.obs_blocks(obs_emb)
        state_obs_rep = self.obs_blocks(obs_emb)
        state_rep = state_obs_rep[:,0:1,:]
        obs_rep = state_obs_rep[:,1:,:]
        action_rep = self.action_encoder(action.float())
        # action_rep = action_emb
        s_a = torch.cat([state_rep, action_rep], dim=1) # concat on n_agent_dim
        seq = self.uncertain_embed(s_a)
        # seq = torch.cat([state_rep, action_rep], dim=1) # v3
        x = self.ac_blocks(seq)
        x = self.relu(x)
        x_cut = x[:,:-1,:]
        x_cut = x_cut + obs_rep
        x = torch.cat([s_a[:,-1:,:].clone(), x_cut], dim=1)
        # x[:,:-1,:] += obs_rep # in-place
        x = x + self.out_proj(x) # residual
        x = self.seq_ln(x)
        logit = self.action_head(x)[:, :-1, :]
        logit = self.relu(logit)
        v_loc = self.value_head(x)[:, :-1, :]
        # out =  self.action_head(x)[:, :-1, :]

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