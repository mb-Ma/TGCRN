import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from script.normalization import dynamic_topK, topK


def season_model(thetas, t):
    '''
    thetas (..., theatas_dim) usually less than 4
    '''
    p = thetas.size()[-1]
    p1, p2 = (p // 2, p // 2) if p % 2 ==0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor(np.array([np.cos(2 * np.pi * i * t) for i in range(p1)])).float()
    s2 = torch.tensor(np.array([np.sin(2 * np.pi * i * t) for i in range(p2)])).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(thetas.device))

def trend_model(thetas, t):
    '''
    thetas is transformerd by Linear layer with X, the last dim is thetas_dim (usuall small).
    t [t1, t2, ,..., tn]    
    [1, t, t^2] \in R^{p, n}
    '''
    p = thetas.size()[-1]
    T = torch.tensor(np.array([t ** i for i in range(p)])).float()
    return thetas.mm(T.to(thetas.device)) # (.., p) (p, n)

class Time2Vec(nn.Module):
    def __init__(self, time_dim):
        super(Time2Vec, self).__init__()

        self.time_dim = time_dim
        self.w_0 = nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.p_0 = nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.W = nn.Parameter(torch.FloatTensor(1, time_dim-1), requires_grad=True)
        self.P = nn.Parameter(torch.FloatTensor(1, time_dim-1), requires_grad=True)
        self.F = torch.sin
    
    def forward(self, inputs):
        """
        inputs [batch, times]
        return [batch, times, dim]
        """
        b_s = inputs.size(0)
        t_s = inputs.size(1)
        # import pdb; pdb.set_trace()
        v1 = self.F(torch.matmul(inputs.view(-1, t_s, 1), self.W) + self.P)
        v0 = torch.matmul(inputs.view(-1, t_s, 1), self.w_0) + self.p_0
        
        return torch.cat([v0, v1], -1)


class time_encoding(nn.Module):
    ''' shift-invariant time encoding kernal
    
    inputs : [N, max_len]
    Returns: 3d float tensor which includes embedding dimension
    '''
    def __init__(self, time_dim):
        super(time_encoding, self).__init__()

        self.effe_numits = time_dim // 2
        self.time_dim = time_dim

        init_freq_base = np.linspace(0, 9, self.effe_numits).astype(np.float32)
        self.cos_freq_var = nn.Parameter((torch.from_numpy(1 / 10.0 ** init_freq_base).float()), requires_grad=False)
        self.sin_freq_var = nn.Parameter((torch.from_numpy(1 / 10.0 ** init_freq_base).float()), requires_grad=False)

        self.beta_var = nn.Parameter((torch.from_numpy(np.ones(time_dim).astype(np.float32)).float()), requires_grad=False)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        inputs = inputs.type(torch.FloatTensor).to(inputs.device)
        inputs = inputs.view(batch_size, seq_len, 1).repeat(1, 1, self.effe_numits)
        
        cos_feat = torch.sin(torch.mul(inputs, self.cos_freq_var.view(1, 1, self.effe_numits)))
        sin_feat = torch.cos(torch.mul(inputs, self.sin_freq_var.view(1, 1, self.effe_numits)))

        freq_feat = torch.cat((cos_feat, sin_feat), dim=-1)

        out = torch.mul(freq_feat, self.beta_var.view(1, 1, self.time_dim))
        return out


class TimeEncode(nn.Module):
    '''
    2020 ICLR
    '''
    def __init__(self, time_dim, factor=5):
        super(TimeEncode, self).__init__()
        self.time_dim = time_dim
        self.factor = factor
        # (0,9) equally divided time_dim 
        self.basis_freq = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim)).float()))
        self.phase = nn.Parameter(torch.zeros(time_dim).float())
    
    def forward(self, ts):
        # ts: [batch, time, node]
        ts = ts.type(torch.FloatTensor).to(ts.device)
        batch_size = ts.size(0)
        seq_len = ts.size(1)
        node_num = ts.size(2)
        ts = ts.view(batch_size, seq_len, node_num, 1)
        map_ts = ts * self.basis_freq.view(1, 1, 1, -1)
        map_ts += self.phase.view(1, 1, 1, -1)
        harmonic = torch.cos(map_ts)

        return harmonic


class ChebGCN(nn.Module):
    """
    ChebNet with node_embeddings [node, dim]
    """
    def __init__(self, in_dim, out_dim, cheb_k, embed_dim):
        super(ChebGCN, self).__init__()
        self.cheb_k = cheb_k
        # learned weights
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, in_dim, out_dim))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, out_dim))
    
    def forward(self, x, node_embeddings):
        # node_embedding [n, d]  or dynamic [b, n, d]
        node_num = node_embeddings.shape[0]

        A = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(A.device), A]
        
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * A, support_set[-1]) - support_set[-2])
        cheb_supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", cheb_supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out

        return x_gconv


class TGCN(nn.Module):
    """
    ChebNet with node_embeddings [batch, node, dim]
    """
    def __init__(self, in_dim, out_dim, cheb_k, embed_dim, period=False):
        super(TGCN, self).__init__()
        self.cheb_k = cheb_k
        # learned weights
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, in_dim, out_dim))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, out_dim))
        self.period = period
        self.init_weights()
    
    def forward(self, x, node_embeddings):
        # x [x;hidden_state]
        # node_embedding [n, d]  or dynamic [b, n, d]
        # add temporal variance 
        bs = x.shape[0]
        node_num = x.shape[1]
        node_embeddings, t, n_t, p = node_embeddings
        t_d = t.size(-1)
        
        # node embedding
        a = torch.mm(node_embeddings, node_embeddings.transpose(0,1)).repeat(x.size(0), 1, 1) # b,n,n
        # time embedding
        a_t = torch.bmm(n_t.view(bs, 1, t_d), t.view(bs, t_d, 1))
        # fusion: add, concat, mlp
        _a = a + a_t
        
        if self.period:
            A = F.softmax(dynamic_topK(F.relu((1+0.3*torch.sigmoid(p))*_a), 10), dim=2)
        else:
            A = F.softmax(F.relu(_a), dim=2)
        
        # A = F.softmax(F.relu(_a), dim=2)
        I = torch.eye(node_num).repeat(x.size(0), 1, 1).to(A.device)
        support_set = [I, A]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * A, support_set[-1]) - support_set[-2])
        
        # node represention and time represenation
        node_embeddings = node_embeddings.repeat(x.size(0), 1, 1)
        node_embeddings = torch.cat((node_embeddings, n_t.unsqueeze(dim=1).repeat(1, node_num, 1)), dim=-1)

        cheb_supports = torch.stack(support_set, dim=1)
        weights = torch.einsum('bnd,dkio->bnkio', node_embeddings, self.weights_pool) #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #B, N, dim_out
        x_g = torch.einsum("bknm,bmc->bknc", cheb_supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,bnkio->bno', x_g, weights) + bias     #b, N, dim_out

        return x_gconv
    
    def init_weights(self):
        nn.init.orthogonal_(self.weights_pool.data)
        nn.init.orthogonal_(self.bias_pool.data)


class GCN(nn.Module):
    """
    GCN with A [batch, node, node]
    apply od matrix and do matrix to gcn
    abandon ChebNet, cos it only operates on undirected graph.
    GCN A=A+I  A{D^-1}XW
    """
    def __init__(self, in_dim, out_dim, deep):
        super(GCN, self).__init__()
        self.init_weights = (nn.Parameter(torch.FloatTensor(in_dim, out_dim)))
        self.weights = (nn.Parameter(torch.FloatTensor(deep-1, out_dim, out_dim)))
        self.deep = deep
        self.out = nn.Linear(deep*out_dim+in_dim, out_dim, bias=True)

    def forward(self, x, A):
        # A = A + I A[batch, N, N]
        I = torch.eye(A.size(1)).repeat(x.size(0), 1, 1).to(A.device)
        A = I+A
        D = A.sum(2).unsqueeze(2).contiguous()
        # AD^{-1}
        _AD = A / D.repeat(1, 1, A.size(2))
        
        state = x
        hidden_out = [state] # [in_dim, out_dim, out_dim, ...]
        
        state = torch.einsum("bmn,bnc->bmc", _AD, state)
        state = torch.einsum('bmc,co->bmo', state, self.init_weights)
        hidden_out.append(state)

        for i in range(1, self.deep):
            state = torch.einsum("bmn,bnc->bmc", _AD, state)
            state = torch.einsum('bmc,co->bmo', state, self.weights[i-1])
            hidden_out.append(state)

        hidden_out = torch.cat(hidden_out, dim=-1) # nodes, (deep+1)*out_dim
        _out = self.out(hidden_out)

        return _out

class AsymGCN(nn.Module):
    """
    process the input two node embedding using GCN
    """
    def __init__(self, in_dim, out_dim, deep):
        super(AsymGCN, self).__init__()
        self.out_dim = out_dim
        self.gcn1 = GCN(in_dim, out_dim, deep)
        self.gcn2 = GCN(in_dim, out_dim, deep)
    
    def forward(self, x, nodes):
        # (batch, 80, 80)
        node1, node2 = nodes
        A = F.softmax(F.relu(torch.bmm(node1, node2.transpose(1, 2))), dim=2)
        A = dynamic_topK(A)

        out = self.gcn1(x, A) + self.gcn2(x, A.transpose(1,2)) + x[..., :self.out_dim]

        return out


class ODGCN(nn.Module):
    """
    process input two matrix
    """
    def __init__(self, in_dim, out_dim, deep):
        super(ODGCN, self).__init__()
        self.out_dim = out_dim
        self.gcn_od = GCN(in_dim, out_dim, deep)
        self.gcn_do = GCN(in_dim, out_dim, deep)
    
    def forward(self, x, A):
        # (batch, 80, 80)
        od, do = A
        out = self.gcn_od(x, od) + self.gcn_do(x, do) + x[..., :self.out_dim]

        return out

class GCGRU(nn.Module):
    """
    RNNCell, 
    """
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_with_station=False, per_flag=False):
        super(GCGRU, self).__init__()
        self.node_num = node_num
        self.time_with_station = time_with_station
        self.hidden_dim = dim_out

        if time_with_station:
            self.gate = TGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim, period=per_flag) # 2*dim_out
            self.update = TGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim, period=per_flag)
        else:
            self.gate = ChebGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim) # 2*dim_out
            self.update = ChebGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        """
        DKGRNCell
        given kg do kgc, missing the periodic modeling, h_t
        1. $$h_t = z_t \odot h_{t-1} + (1-z_t) \odot \hat{h_t}$$ 
        2. $$\hat{h_t} = tanh(A[X_{:,t}, r \odot h_{t-1}]EW_{\hat{h}} + Eb_{\hat{h}})$$ 
        3. r_t = sigmoid(A[X_{:,t}, h_{t-1}]EW_r + Eb_r)
        4. z_t = sigmoid(A[X_{:,t}, h_{t-1}]EW_z + Eb_z)
        5. graph convolution
        """
        #x: B, num_nodes, input_dim(x, time_embedding)
        if self.time_with_station:
            node_embeddings, n_t, t, p = node_embeddings
        #state: B, num_nodes, hidden_dim : h_{t-1}
        state = state.to(x.device)
        # 1. [X:,t, h_{t-1}]
        input_and_state = torch.cat((x, state), dim=-1) # [b, n, dim_in_+dim_out]
        # 2. z_t   gate mechanism
        if self.time_with_station:
            z_r = torch.sigmoid(self.gate(input_and_state, (node_embeddings, t, n_t, p))) # output dim: b, n, dim_out
        else:
            z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1) # z[b, n, dim_out]， r[b, n, dim_out]
        # 3. 
        candidate = torch.cat((x, r*state), dim=-1)
        # 5. update mechanism \hat{h_t}
        if self.time_with_station:
            _h = torch.tanh(self.update(candidate, (node_embeddings, t, n_t, p)))
        else:
            _h = torch.tanh(self.update(candidate, node_embeddings))
        # 6. final hidden state
        h = z*state + (1-z)*_h
        return h
    
    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class GRUCell(nn.Module):
    """
    RNNCell, 
    """
    def __init__(self, dim_in, dim_out):
        super(GRUCell, self).__init__()
        self.hidden_dim = dim_out
        self.W_z_r = nn.Parameter(torch.FloatTensor(dim_in+dim_out, 2*dim_out))
        self.b_z_r = nn.Parameter(torch.FloatTensor(2*dim_out))
        self.W_h = nn.Parameter(torch.FloatTensor(dim_in+dim_out, dim_out))
        self.b_h = nn.Parameter(torch.FloatTensor(dim_out))

    def forward(self, x, state):
        """
        DKGRNCell
        given kg do kgc, missing the periodic modeling, h_t
        1. $$h_t = z_t \odot h_{t-1} + (1-z_t) \odot \hat{h_t}$$ 
        2. $$\hat{h_t} = tanh(A[X_{:,t}, r \odot h_{t-1}]EW_{\hat{h}} + Eb_{\hat{h}})$$ 
        3. r_t = sigmoid([X_{:,t}, h_{t-1}]W_r + b_r)
        4. z_t = sigmoid([X_{:,t}, h_{t-1}]W_z + b_z)
        5. graph convolution
        """
        #x: B, num_nodes, input_dim(x, time_embedding)
        #state: B, num_nodes, hidden_dim : h_{t-1}
        state = state.to(x.device)
        # 1. [X:,t, h_{t-1}]
        input_and_state = torch.cat((x, state), dim=-1) # [b, n, dim_in_+dim_out]
        # 2. z_t   gate mechanism
        z_r = torch.einsum("io,bni->bno", self.W_z_r, input_and_state) + self.b_z_r
        z, r = torch.split(z_r, self.hidden_dim, dim=-1) # z[b, n, dim_out]， r[b, n, dim_out]
        # 3. 
        candidate = torch.cat((x, r*state), dim=-1)
        # 5. update mechanism \hat{h_t}
        _h = torch.einsum("io,bni->bno", self.W_h, candidate) + self.b_h
        # 6. final hidden state
        h = z*state + (1-z)*_h
        return h
    
    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)



if __name__=='__main__':
    te = time_encoding(64)
    t1 = torch.FloatTensor([[[0.]]])
    t2 = torch.FloatTensor([[[1.]]])
    t3 = torch.FloatTensor([[[2.]]])

    t1_embed = te(t1)
    t2_embed = te(t2)
    t3_embed = te(t3)
    print(t1_embed[0,0,0,:])
    print(t2_embed[0,0,0,:])
    print(t3_embed[0,0,0,:])
    print(torch.dot(t1_embed[0,0,0,:], t2_embed[0,0,0,:]))
    print(torch.dot(t2_embed[0,0,0,:], t3_embed[0,0,0,:]))