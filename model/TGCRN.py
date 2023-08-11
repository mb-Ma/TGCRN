import torch
import torch.nn as nn
from model.GCGRU import GCGRU, GRUCell, TimeEncode, time_encoding, Time2Vec
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, node_num, in_dim, out_dim, cheb_k, embed_dim, num_layers=1, od_flag=False, time_station=False, g_d="symm", per_flag=False):
        super(Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DKGRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = in_dim
        self.od_flag = od_flag
        self.g_d = g_d
        self.per_flag = per_flag
        self.time_station = time_station
        self.num_layers = num_layers
        self.gcgru_cells = nn.ModuleList()

        self.gcgru_cells.append(GCGRU(node_num, in_dim, out_dim, cheb_k, embed_dim, time_station, per_flag))
        for _ in range(1, num_layers):
            self.gcgru_cells.append(GCGRU(node_num, out_dim, out_dim, cheb_k, embed_dim, time_station, per_flag))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        
        if self.od_flag:
            od, do = node_embeddings
        
        if self.time_station:
            if self.g_d == "asym":
                node1, node2 = node_embeddings
            else:
                node_embeddings, t_embed = node_embeddings
            
        assert x.shape[2] == self.node_num
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []

        # tmp addding
        if self.per_flag:
            p1 = torch.tanh(torch.bmm(x[..., 0].transpose(1,2), x[..., 0]))
            p2 = torch.tanh(torch.bmm(x[..., 1].transpose(1,2), x[..., 1]))
            p = (p1 + p2) / 2.
        else:
            p = None
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            # encoding
            for t in range(seq_length):
                
                if self.od_flag:
                    state = self.gcgru_cells[i](current_inputs[:, t, :, :], state, (od[:,t,:,:], do[:,t,:,:]))
                elif self.time_station:
                    if self.g_d == 'asym':
                        state = self.gcgru_cells[i](current_inputs[:, t, :, :], state, (node1[:,t,:,:], node2[:,t,:,:]))
                    else:
                        if t == 0:
                            state = self.gcgru_cells[i](current_inputs[:, t, :, :], state, (node_embeddings, t_embed[:,0,:], t_embed[:,0,:], p))
                        else:
                            state = self.gcgru_cells[i](current_inputs[:, t, :, :], state, (node_embeddings, t_embed[:,t,:], t_embed[:,t-1,:], p))
                else:
                    state = self.gcgru_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            current_inputs = torch.stack(inner_states, dim=1) #（B,T,N,Hidden_dim）
            output_hidden.append(state) # (B, N, Hidden_dim)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.gcgru_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)


class Decoder(nn.Module):
    def __init__(self, node_num, horizon, in_dim, rnn_units, out_dim, cheb_k, embed_dim, num_layers=1, od_flag=False, time_station=False, g_d='asym',per_flag=False):
        super(Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DKGRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = in_dim # run_units plus time_embedding
        self.out_dim = out_dim # output dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.od_flag = od_flag
        self.g_d = g_d
        self.per_flag = per_flag
        self.time_station = time_station
        self.gcgru_cells = nn.ModuleList()
        
        self.gcgru_cells.append(GCGRU(node_num, in_dim, rnn_units, cheb_k, embed_dim, time_station, per_flag))
        for _ in range(1, num_layers):
            self.gcgru_cells.append(GCGRU(node_num, rnn_units, rnn_units, cheb_k, embed_dim, time_station, per_flag))
        
        self.end_out = nn.Linear(rnn_units, out_dim)
        self.mlp1 = nn.Linear(in_dim,1)

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        if self.time_station:
            if self.g_d == "asym":
                node1, node2 = node_embeddings
            else:
                node_embeddings, t_embed = node_embeddings

        assert x.shape[2] == self.node_num
        current_inputs = x
        output_hidden = []
                    
        # tmp addding
        if self.per_flag:
            tmp = self.mlp1(current_inputs).view(current_inputs.shape[0], current_inputs.shape[1], x.shape[2])
            p = torch.tanh(torch.bmm(tmp.transpose(1,2), tmp))
        else:
            p = None

        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []

            for t in range(self.horizon): 
                # node_embedding
                if self.time_station:
                    if self.g_d == "asym":
                        state = self.gcgru_cells[i](current_inputs[:, t, :, :], state, (node1[:,t,:,:], node2[:,t,:,:]))
                    else:
                        if t == 0:
                            state = self.gcgru_cells[i](current_inputs[:, t, :, :], state, (node_embeddings, t_embed[:,0,:], t_embed[:,0,:], p))
                        else:
                            state = self.gcgru_cells[i](current_inputs[:, t, :, :], state, (node_embeddings, t_embed[:,t,:], t_embed[:,t-1,:], p))
                else:
                    state = self.gcgru_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            current_inputs = torch.stack(inner_states, dim=1) # （B,T,N,Hidden_dim）
            output_hidden.append(state) # (B, N, Hidden_dim)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        out = self.end_out(current_inputs)
        return out
    
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.gcgru_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)


class Decoder_WOG(nn.Module):
    def __init__(self, node_num, horizon, in_dim, rnn_units, out_dim, cheb_k, embed_dim, num_layers=1):
        super(Decoder_WOG, self).__init__()
        assert num_layers >= 1, 'At least one DKGRNN layer in the Encoder.'
        self.input_dim = in_dim # run_units+time_embedding
        self.out_dim = out_dim # output dim
        self.horizon = horizon
        self.node_num = node_num
        self.num_layers = num_layers
        self.gru_cells = nn.ModuleList()
        
        # the input dim of first layer is different from the reminder layers
        self.gru_cells.append(GRUCell(in_dim, rnn_units))
        for _ in range(1, num_layers):
            self.gru_cells.append(GRUCell(rnn_units, rnn_units))
        
        self.end_out = nn.Linear(rnn_units, out_dim)

    def forward(self, x, init_state):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num
        current_inputs = x
        output_hidden = []

        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            
            for t in range(self.horizon):
                state = self.gru_cells[i](current_inputs[:, t, :, :], state)
                inner_states.append(state)
            current_inputs = torch.stack(inner_states, dim=1) # （B,T,N,Hidden_dim）
            output_hidden.append(state) # (B, N, Hidden_dim)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        out = self.end_out(current_inputs)
        return out



class TGCRN(nn.Module):
    def __init__(self, args):
        super(TGCRN, self).__init__()
        self.args = args
        self.num_nodes = args.num_nodes
        self.input_dim = args.input_dim # input dim
        self.time_dim = args.time_dim # time embedding dim
        self.rnn_units = args.rnn_units # hidden state dim
        self.output_dim = args.output_dim # output dim
        self.horizon = args.horizon # output len
        self.num_layers = args.num_layers # network layer
        self.embed_dim = args.embed_dim # node embedding dim
        self.cheb_k = args.cheb_k

        # random representation for node
        if args.graph_direction == "symm":
            self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim), requires_grad=True)
        if args.graph_direction == "asym":
            self.node_embeddings1 = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim), requires_grad=True)
            self.node_embeddings2 = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim), requires_grad=True)
        
        # embedding weight
        self.node_weight = nn.Parameter(torch.randn(args.embed_dim, args.embed_dim), requires_grad=True)
        self.time_weight = nn.Parameter(torch.randn(args.time_dim, args.time_dim), requires_grad=True)
        

        # time_embedding
        if args.time_embedding:
            # self.time_embed = nn.Embedding(num_embeddings=args.num_time, embedding_dim=args.time_dim) # Embedding
            # self.time_embed = TimeEncode(args.time_dim) # 2020 ICLR
            self.time_embed = time_encoding(args.time_dim) # 2019 NIPS
            # self.time_embed = Time2Vec(args.time_dim) # time2vec

        # node representation with time info
        if args.time_station:
            self.embed_dim += self.time_dim

        # encoder
        self.encoder = Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k,
                                self.embed_dim, self.num_layers, args.od_flag, args.time_station, args.graph_direction, args.period)
        
        # decoder
        if args.Seq_Dec:
            if args.time_embedding:
                # self.input_dim = self.rnn_units + self.time_dim
                self.input_dim = self.rnn_units
            else:
                self.input_dim = self.rnn_units
    
            if args.od_flag:
                self.decoder = Decoder_WOG(self.num_nodes, self.horizon, self.input_dim, self.rnn_units, self.output_dim,  self.cheb_k,
                                    self.embed_dim, self.num_layers)
            else:
                self.decoder = Decoder(self.num_nodes, self.horizon, self.input_dim, self.rnn_units, self.output_dim,  self.cheb_k,
                                    self.embed_dim, self.num_layers, args.od_flag, args.time_station, args.graph_direction, args.period)

        else: # 一次解码
            self.decoder = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.rnn_units), bias=True)
        
    def forward(self, _input):
        """
        if time_embedding is true:
            input: X [batch, T, node, feat], X_time [batch, T], Y_time [batch, T]
        else:
            input: X
        """
        
        if self.args.time_embedding and self.args.constrative_time and self.training:
            _input, x_time, y_time, cons_time= _input
        elif self.args.time_embedding:
            if self.args.od_flag:
                _input, x_time, y_time, od, do = _input
            else:
                _input, x_time, y_time = _input
        else:
            pass

        if self.args.time_embedding:   
            # [batch, time, dim]
            x_time_feat = self.time_embed(x_time)
            y_time_feat = self.time_embed(y_time)

        # add node embedding constrains. 
        node = F.tanh(torch.mm(self.node_embeddings, self.node_weight))

        # init hidden states
        init_state = self.encoder.init_hidden(_input.shape[0])

        # get the last hidden states from the last layer and every layer
        if self.args.od_flag:
            output, _ = self.encoder(_input, init_state, (od, do)) # output: [B, T, N, hidden]
        elif self.args.time_station:
            # graph direction
            if self.args.graph_direction == "symm":
                # change <[e1||t], [e2||t]> --> <e1, e2>+<t, t-1>
                # _node_embed = self.stat_with_time(self.node_embeddings, x_time_feat)
                output, _ = self.encoder(_input, init_state, (node, x_time_feat))
            if self.args.graph_direction == "asym":
                node1 = self.stat_with_time(self.node_embeddings1, x_time_feat)
                node2 = self.stat_with_time(self.node_embeddings2, x_time_feat)
                output, _ = self.encoder(_input, init_state, (node1, node2))
        else:
            output, _ = self.encoder(_input, init_state, self.node_embeddings) # output: [B, T, N, hidden]
    
        # decoding
        if self.args.Seq_Dec:
            init_state = self.decoder.init_hidden(_input.shape[0])
            
            if self.args.od_flag:
                outs = self.decoder(output, _)
            
            if self.args.time_station:
                if self.args.graph_direction == "symm":
                    # _node_embed = self.stat_with_time(self.node_embeddings, x_time_feat)
                    outs = self.decoder(output, _, (node, y_time_feat))
                if self.args.graph_direction == "asym":
                    node1 = self.stat_with_time(self.node_embeddings1, y_time_feat)
                    node2 = self.stat_with_time(self.node_embeddings2, y_time_feat)
                    outs = self.decoder(output, _, (node1, node2))
            else:
                outs = self.decoder(output, _, self.node_embeddings)
        else:
            output = output[:, -1:, :, :]                                   #B, 1, N, hidden
            #CNN based predictor
            output = self.decoder((output))                         #B, T*C, N, 1
            output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_nodes)
            outs = output.permute(0, 1, 3, 2)                             #B, T, N, C    

        if self.args.constrative_time and self.training:
            cons_time_embed = self.time_embed(cons_time)
            ratios = []
            
            for i in range(1,4):
                embed_diff = torch.mean((cons_time_embed[:,0,:] - cons_time_embed[:,i,:]) ** 2, dim=-1, keepdim=True) # 16 x 1
                distance_diff = torch.abs(cons_time[:,0] - cons_time[:,i])  # 16 x 1
                ratio = embed_diff / distance_diff.float().clamp_(1e-6)
                ratios.append(ratio)
            return outs, ratios

        return outs
    
    def stat_with_time(self, node_embeddings, time_feat):
        # node_embedding [n, dim], time [batch, T, dim] -> [batch, T, N, dim]
        _node_embed = node_embeddings.repeat(time_feat.size(0), time_feat.size(1), 1, 1)
        _node_embed = torch.cat((_node_embed, time_feat), dim=-1)

        return _node_embed
