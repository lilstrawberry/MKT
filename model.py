from sklearn import metrics
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import glo

def getAttention(query, key, value, mask, fn):
    d = query.shape[-1]
    attn = torch.matmul(query, key.transpose(-1, -2))
    # / math.sqrt(d)
    now_attn = torch.masked_fill(attn, mask, -1e9)
    now_attn = torch.softmax(now_attn, dim=-1)
    # now_attn = fn(now_attn)
    res = torch.matmul(now_attn, value)
    return now_attn, res

class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim, p):
        super(GCNConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w = nn.Parameter(torch.rand((in_dim, out_dim)))
        nn.init.xavier_uniform_(self.w)

        self.convert = nn.Parameter(torch.rand((in_dim, out_dim)))
        nn.init.xavier_uniform_(self.convert)

        self.b = nn.Parameter(torch.rand((out_dim)))
        nn.init.zeros_(self.b)

        self.dropout = nn.Dropout(p=p)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = torch.matmul(x, self.w)
        x = torch.sparse.mm(adj.float(), x)
        x = x + self.b
        return x

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_feature, out_feature, alpha, dropout):
        super(GraphAttentionLayer, self).__init__()

        self.out_feature = out_feature

        self.W = nn.Parameter(torch.empty((in_feature, out_feature)))
        self.a = nn.Parameter(torch.empty((2 * out_feature, 1)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.C = nn.Parameter(torch.empty((in_feature, out_feature)))
        nn.init.xavier_uniform_(self.C.data, gain=1.414)

        self.dropout = nn.Dropout(p=dropout)
        self.leakRelu = nn.LeakyReLU(alpha)

    def prepare(self, h):
        # h: n out_feature

        h_i = torch.matmul(h, self.a[:self.out_feature, :])
        # n 1
        h_j = torch.matmul(h, self.a[self.out_feature:, :])
        # n 1
        e = h_i + h_j.T
        # n n
        e = self.leakRelu(e)
        return e

    def forward(self, x):
        #   x: n in_feature
        # adj: n n
        adj = glo.get_value('gat_matrix')

        batch_size = 10000
        device = x.device
        num_nodes = x.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        indices = torch.arange(0, num_nodes).to(device)

        h = torch.matmul(x, self.W)
        # n out_feature
        dd = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]

            h_i = torch.matmul(h[mask], self.a[:self.out_feature, :])  # mask 1
            h_j = torch.matmul(h, self.a[self.out_feature:, :])  # n 1
            e = h_i + h_j.T  # mask n
            e = self.leakRelu(e)

            now_mask = (adj[mask] <= 0)  # mask n
            attn = torch.masked_fill(e, now_mask, -1e9)

            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            # mask n
            res = torch.matmul(attn, h)

            # res = F.elu(res) # mask n
            # res = self.dropout(res)

            dd.append(res)
        dd = torch.vstack(dd)

        return dd

class ContrastLoss(nn.Module):
    def __init__(self, hidden, tau, contrast_lamda=0.5):
        super(ContrastLoss, self).__init__()

        self.tau = tau
        self.contrast_lamda = contrast_lamda

        self.project_linear = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden)
        )
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.414)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        res = torch.matmul(z1, z2.transpose(-1, -2))
        res = torch.exp(res / self.tau)
        return res

    def batched_semi_loss(self, z1, z2, batch_size, pro_skill_embed, index=None):

        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        indices = torch.arange(0, num_nodes).to(device)

        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]

            if index is None:
                need_contrast_matrix = glo.get_value('gat_matrix')[mask]  # batch N
            else:
                select_index = index[i * batch_size:(i + 1) * batch_size]
                need_contrast_matrix = glo.get_value('gat_matrix')[select_index]  # batch N

            now_skill = pro_skill_embed[mask]

            now_vec = z1[mask]  # batch d
            fx_sim = self.sim(now_vec, z2)  # batch, N

            now_to_skill = (now_skill * now_vec).sum(dim=-1, keepdims=True)
            now_to_skill = torch.exp(now_to_skill / self.tau)

            fenzi = now_to_skill + (fx_sim * need_contrast_matrix).sum(dim=-1, keepdims=True)
            fenmu = now_to_skill + fx_sim.sum(dim=-1, keepdims=True)
            #             fenzi = (fx_sim * need_contrast_matrix).sum(dim=-1, keepdims=True)
            #             fenmu = fx_sim.sum(dim=-1, keepdims=True)

            sc_loss = -torch.log(fenzi / (fenmu + 1e-8))

            #             fx_sim = (fx_sim + now_to_skill) / (now_to_skill + torch.sum(fx_sim, dim=-1, keepdim=True) + 1e-8)
            #             sc_loss = -torch.log((fx_sim * need_contrast_matrix).sum(-1) + 1e-8)

            losses.append(sc_loss)

        res_loss = torch.cat(losses).mean()

        return res_loss

    def forward(self, x1, x2, pro_skill_embed, index=None, sele_idx=None, rest_idx=None, gen_skill=None, gen_pro=None):

        contrast_loss = self.batched_semi_loss(x1, x2, 10000, pro_skill_embed, index)

        return contrast_loss


class MKT(nn.Module):
    def __init__(self, skill_max, pro_max, embed, state_d, p):
        super(MKT, self).__init__()

        self.skill_max = skill_max
        self.pro_max = pro_max

        # self.gat = GraphAttentionLayer(d, d, 0.2, p)

        self.pro_embed = nn.Parameter(torch.rand(pro_max, embed))
        nn.init.xavier_uniform_(self.pro_embed)

        self.skill_embed = nn.Parameter(torch.rand(skill_max, embed))
        nn.init.xavier_uniform_(self.skill_embed)

        self.var = nn.Parameter(torch.rand(pro_max, embed))
        self.change = nn.Parameter(torch.rand(pro_max, 1))

        self.pos_embed = nn.Parameter(torch.rand(glo.get_value('max_seq'), embed))
        nn.init.xavier_uniform_(self.pos_embed)

        state_embed = state_d

        self.pro_state = nn.Parameter(torch.rand(pro_max, state_embed))
        self.skill_state = nn.Parameter(torch.rand(skill_max, state_embed))
        self.time_state = nn.Parameter(torch.rand(glo.get_value('max_seq'), state_embed))
        self.all_state = nn.Parameter(torch.rand(1, state_embed))

        self.all_forget = nn.Sequential(
            nn.Linear(2 * state_embed, state_embed),
            nn.ReLU(),
            nn.Linear(state_embed, state_embed),
            nn.Sigmoid()
        )

        self.skill_forget = nn.Sequential(
            nn.Linear(2 * state_embed + 1 * state_embed, state_embed),
            nn.ReLU(),
            nn.Linear(state_embed, state_embed),
            nn.Sigmoid()
        )

        self.pro_forget = nn.Sequential(
            nn.Linear(2 * state_embed + 2 * state_embed, state_embed),
            nn.ReLU(),
            nn.Linear(state_embed, state_embed),
            nn.Sigmoid()
        )

        self.now_out = nn.Sequential(
            nn.Linear(3 * state_embed + 2 * embed, 2 * state_embed),
            nn.Tanh(),
            nn.Linear(2 * state_embed, state_embed),
            nn.Tanh(),
            nn.Linear(state_embed, 1)
        )

        self.out1_get = nn.Linear(3 * state_embed + 2 * embed, 3 * state_embed)
        self.state_embed = state_embed

        self.update_pro = nn.Sequential(
            nn.Linear(2 * embed + 0 * state_embed, state_embed),
            nn.Tanh(),
            nn.Linear(state_embed, state_embed),
            nn.Tanh()
        )
        self.update_skill = nn.Sequential(
            nn.Linear(2 * embed + 0 * state_embed, state_embed),
            nn.Tanh(),
            nn.Linear(state_embed, state_embed),
            nn.Tanh()
        )
        self.update_all = nn.Sequential(
            nn.Linear(2 * embed + 0 * state_embed, state_embed),
            nn.Tanh(),
            nn.Linear(state_embed, state_embed),
            nn.Tanh()
        )

        self.aug_pro_state = nn.Sequential(
            nn.Linear(1 * state_embed + 1 * embed, state_embed),
            nn.ReLU(),
            nn.Linear(state_embed, state_embed),
            nn.Tanh()
        )

        self.aug_skill_state = nn.Sequential(
            nn.Linear(1 * state_embed + 2 * embed, state_embed),
            nn.ReLU(),
            nn.Linear(state_embed, state_embed),
            nn.Tanh()
        )

        self.clas = nn.Sequential(
            nn.Linear(embed, embed),
            nn.ReLU(),
            nn.Linear(embed, skill_max)
        )

        d = embed

        self.skill_proj = nn.Linear(d, d)

        self.gcn = GCNConv(d, d, p)

        self.contrast = ContrastLoss(d, 0.8)

        self.alpha = nn.Parameter(torch.rand(1))

        self.ans_embed = nn.Embedding(2, d)
        self.lstm = nn.LSTM(d, d, batch_first=True)

        self.out = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1)
        )

        self.key1_linear = nn.Linear(d, 2 * d)
        self.key2_linear = nn.Linear(d, 2 * d)
        self.key3_linear = nn.Linear(d, 2 * d)

        self.state1_linear = nn.Linear(state_embed, state_embed)
        self.state2_linear = nn.Linear(state_embed, state_embed)
        self.state3_linear = nn.Linear(state_embed, state_embed)

        self.now_obtain = nn.Sequential(
            nn.Linear(2 * d, state_embed),
            nn.Tanh(),
            nn.Linear(state_embed, state_embed),
            nn.Tanh()
        )

        self.f1_get = nn.Linear(4 * state_embed, 3 * state_embed)

        self.dropout = nn.Dropout(p=p)
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def mixup_data(self, x, alpha=1.0):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.shape[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x, index, lam

    def compute_sim_loss(self, x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return torch.sum(x * y, dim=-1).mean()

    def clas_loss(self, pro, skill):
        pro_clas = self.clas(pro)
        pro_clas = pro_clas.view(-1, pro_clas.shape[-1])
        skill = skill.view(-1)
        cri = nn.CrossEntropyLoss()
        return cri(pro_clas, skill)

    def forward(self, last_problem, pos_problem, neg_problem,
                last_ans, last_skill, next_skill,
                neg_skill, next_problem, next_ans, perb=None):

        device = last_problem.device

        seq = last_problem.shape[1]
        batch = last_problem.shape[0]

        # pro2skill pro skill

        pro_embed = self.pro_embed

        mixed_x, index, lam = self.mixup_data(pro_embed)
        other_x = F.embedding(index, pro_embed)

        pro2skill_index = glo.get_value('pro2skill_index')

        pro_skill_embed = F.embedding(pro2skill_index, self.skill_embed)
        other_skill_index = torch.index_select(pro2skill_index, 0, index)
        other_skill_embed = F.embedding(other_skill_index, self.skill_embed)

        contrast_mix = self.contrast(pro_embed, pro_embed, pro_skill_embed)

        last_pro_rasch = F.embedding(last_problem, pro_embed)
        next_pro_rasch = F.embedding(next_problem, pro_embed)
        next_skill_embed = F.embedding(next_skill, self.skill_embed)

        l2 = glo.get_value('regist_l2')

        contrast_loss = contrast_mix * l2

        X = last_pro_rasch + self.ans_embed(last_ans)
        next_X = next_pro_rasch + self.ans_embed(next_ans)

        res_p = []
        last_pro_time = torch.zeros((batch, self.pro_max)).to(device).long()  # batch pro
        last_skill_time = torch.zeros((batch, self.skill_max)).to(device).long()  # batch skill
        last_all_time = torch.ones((batch)).to(device).long()

        time_embed = self.time_state  # seq d
        all_gap_embed = F.embedding(last_all_time, time_embed)  # batch d

        skill_state = self.skill_state.unsqueeze(0).repeat(batch, 1, 1)  # batch skill d
        pro_state = self.pro_state.unsqueeze(0).repeat(batch, 1, 1)  # batch pro d
        all_state = self.all_state.repeat(batch, 1)  # batch d
        batch_indices = torch.arange(batch).to(device)

        for now_step in range(seq):
            now_pro = next_problem[:, now_step]  # batch
            now_skill = next_skill[:, now_step]  # batch

            now_pro_embed = next_pro_rasch[:, now_step]
            now_skill_embed = next_skill_embed[:, now_step]

            pro_time_gap = now_step - last_pro_time[batch_indices, now_pro]  # batch
            skill_time_gap = now_step - last_skill_time[batch_indices, now_skill]  # batch

            pro_time_gap_embed = F.embedding(pro_time_gap, time_embed)  # batch d
            skill_time_gap_embed = F.embedding(skill_time_gap, time_embed)  # batch d

            now_pro_state = pro_state[batch_indices, now_pro]  # batch d
            now_skill_state = skill_state[batch_indices, now_skill]  # batch d
            now_all_state = all_state  # batch d

            forget_now_all_state = now_all_state * self.all_forget(
                self.dropout(torch.cat([now_all_state, all_gap_embed], dim=-1)))
            forget_now_skill_state = now_skill_state * self.skill_forget(
                self.dropout(torch.cat([now_skill_state, skill_time_gap_embed, forget_now_all_state], dim=-1)))
            forget_now_pro_state = now_pro_state * self.pro_forget(self.dropout(
                torch.cat([now_pro_state, pro_time_gap_embed, forget_now_all_state, forget_now_skill_state], dim=-1)))

            #             forget_now_all_state = now_all_state
            #             forget_now_skill_state = now_skill_state
            #             forget_now_pro_state = now_pro_state

            #             forget_now_all_state = torch.clamp(forget_now_all_state, -2, 2)
            #             forget_now_skill_state = torch.clamp(forget_now_skill_state, -2, 2)
            #             forget_now_pro_state = torch.clamp(forget_now_pro_state, -2, 2)

            now_do_state = torch.cat([forget_now_pro_state, forget_now_skill_state, forget_now_all_state], dim=-1)
            # batch 3d

            #             query = torch.cat([now_pro_embed, now_skill_embed], dim=-1) # batch 2d
            #             key1 = self.key1_linear(forget_now_pro_state)  # batch 2d
            #             key2 = self.key2_linear(forget_now_skill_state) # batch 2d
            #             key3 = self.key3_linear(forget_now_all_state)   # batch 2d
            #             attn1 = (query * key1).sum(-1).unsqueeze(-1) # batch 1
            #             attn2 = (query * key2).sum(-1).unsqueeze(-1) # batch 1
            #             attn3 = (query * key3).sum(-1).unsqueeze(-1) # batch 1
            #             attn = torch.cat([attn1, attn2, attn3], dim=-1) # batch 3
            #             attn = torch.softmax(attn, dim=-1) # batch 3
            #             now_do_state = torch.cat([attn[:, 0].unsqueeze(-1) * self.state1_linear(forget_now_pro_state), attn[:, 1].unsqueeze(-1) * self.state2_linear(forget_now_skill_state), attn[:, 2].unsqueeze(-1) * self.state3_linear(forget_now_all_state)], dim=-1)

            now_input = torch.cat([now_do_state, now_pro_embed, now_skill_embed], dim=-1)

            now_input_attn = self.out1_get(now_input)
            fa = now_input_attn[:, :self.state_embed].unsqueeze(-1)
            fb = now_input_attn[:, self.state_embed:-self.state_embed].unsqueeze(-1)
            fc = now_input_attn[:, -self.state_embed:].unsqueeze(-1)

            fd = torch.cat([fa, fb, fc], dim=-1)
            fd = torch.softmax(fd, dim=-1)
            now_input_attn = fd.transpose(-1, -2)

            pro_attn_state = now_input_attn[:, 1]
            skill_attn_state = now_input_attn[:, -1]
            all_attn_state = now_input_attn[:, 0]

            now_input = torch.cat([pro_attn_state * forget_now_pro_state, skill_attn_state * forget_now_skill_state,
                                   all_attn_state * forget_now_all_state], dim=-1)
            now_input = torch.cat([now_input, now_pro_embed, now_skill_embed], dim=-1)

            now_output = torch.sigmoid(self.now_out(self.dropout(now_input)).squeeze(-1))  # batch
            res_p.append(now_output)

            last_pro_time[batch_indices, now_pro] = now_step
            last_skill_time[batch_indices, now_skill] = now_step

            now_X = next_X[:, now_step]  # batch d

            to_get = self.now_obtain(self.dropout(torch.cat([now_X, now_skill_embed], dim=-1)))  # batch state

            now_concat = torch.cat([to_get, forget_now_pro_state, forget_now_skill_state, forget_now_all_state], dim=-1)

            now_concat_attn = self.f1_get(now_concat)
            fa = now_concat_attn[:, :self.state_embed].unsqueeze(-1)
            fb = now_concat_attn[:, self.state_embed:-self.state_embed].unsqueeze(-1)
            fc = now_concat_attn[:, -self.state_embed:].unsqueeze(-1)
            fd = torch.cat([fa, fb, fc], dim=-1)
            fd = torch.softmax(fd, dim=-1)
            now_concat_attn = fd.transpose(-1, -2)

            pro_get = now_concat_attn[:, 0]
            skill_get = now_concat_attn[:, 1]
            all_get = now_concat_attn[:, -1]

            pro_state[batch_indices, now_pro] = forget_now_pro_state + pro_get * to_get
            skill_state[batch_indices, now_skill] = forget_now_skill_state + skill_get * to_get
            all_state = forget_now_all_state + all_get * to_get

        P = torch.vstack(res_p).T

        return P, contrast_loss