import torch
import numpy as np
from script.normalization import NScaler, MinMaxScaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
import pickle
import os
import sys
from script.TimeFeature import build_time_embedding, build_real_time
from datetime import datetime, timedelta
import json
import random
        

def load_dataset(data_dir, normalize, batch_size, args, if_time_embedding=False):
    """
    train: (1188, 4, 80, 2), time: (1188, 4)
    """
    data = {}
    for category in ["train", "val", "test"]:
        cat_data = load_pickle(os.path.join(data_dir, category + '.pkl'))
        data['x_' + category] = cat_data['x']
        data['xtime_' + category] = cat_data['xtime']
        data['y_' + category] = cat_data['y']
        data['ytime_' + category] = cat_data['ytime']
        if args.od_flag:
            data['xod_' + category] = cat_data['xod'] # samples, t, n, n
            data['xdo_' + category] = np.transpose(data['xod_' + category], (0,1,3,2)) # OD^T
    if args.data == 'HZ':
        # reorganize dataset
        data['x_test'] = np.concatenate([data['x_val'][-66:], data['x_test']], 0)
        data['y_test'] = np.concatenate([data['y_val'][-66:], data['y_test']], 0)
        data['xtime_test'] = np.concatenate([data['xtime_val'][-66:], data['xtime_test']], 0)
        data['ytime_test'] = np.concatenate([data['ytime_val'][-66:], data['ytime_test']], 0)
        #import pdb; pdb.set_trace()
        data['x_val'] = data['x_val'][:-66]
        data['y_val'] = data['y_val'][:-66]
        data['xtime_val'] = data['xtime_val'][:-66]
        data['ytime_val'] = data['ytime_val'][:-66]

        data['x_val'] = np.concatenate([data['x_train'][-66:], data['x_val']], 0)
        data['y_val'] = np.concatenate([data['y_train'][-66:], data['y_val']], 0)
        data['xtime_val'] = np.concatenate([data['xtime_train'][-66:], data['xtime_val']], 0)
        data['ytime_val'] = np.concatenate([data['ytime_train'][-66:], data['ytime_val']], 0)

        data['x_train'] = data['x_train'][:-66]
        data['y_train'] = data['y_train'][:-66]
        data['xtime_train'] = data['xtime_train'][:-66]
        data['ytime_train'] = data['ytime_train'][:-66]
    
    # Normalization
    if normalize == "std":
        scaler = StandardScaler(mean=data['x_train'].mean(),
                            std=data['x_train'].std())
        data["scaler"] = scaler

    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category])

        if args.od_flag:
            # od normalization
            data['xod_' + category] = data['xod_' + category] / np.tile((data['xod_' + category].sum(axis=3) + 1e-10)[..., np.newaxis], (1, 1, 1, args.num_nodes))
            data['xdo_' + category] = data['xdo_' + category] / np.tile((data['xdo_' + category].sum(axis=3) + 1e-10)[..., np.newaxis], (1, 1, 1, args.num_nodes))
 
    # build time feature, weekdays timeslot embedding and weekends/holidays timeslot embedding (不是时间点嵌入，而是时间段嵌入)
    if if_time_embedding:
        # metro data: 5:30-6:15 (4), 5:30-23:15 (18*4=72), 5:30-11:30 (73)
        # bike data: 00:00, 00:30, ..., 23:30, (48)
        time = []
        if args.data in ["HZ","SH"]:
        # for metro
            base_time = datetime(2019, 1, 1, 5, 30)
            for i in range(73):
                time.append("{}:{}".format(str(base_time.hour), str(base_time.minute)))
                base_time = base_time + timedelta(minutes=15)
        
        if args.data in ["taxi", "bike"]:
            base_time = datetime(2019, 1, 1, 0, 0)
            for i in range(48):
                time.append("{}:{}".format(str(base_time.hour), str(base_time.minute)))
                base_time = base_time + timedelta(minutes=30)
            
        # make it sequential flatten
        time2id = {}
        for t in time:
            time2id["0:"+t] = len(time2id) # workdays 48/73
        for t in time:
            time2id["1:"+t] = len(time2id) # weekend and holiday
        
        id2time = {v:k for k,v in time2id.items()}
        with open('../data/{}/time_id.json'.format(data_dir[8:]), 'w') as f:
            json.dump([time2id, id2time], f)
        
        
        for category in ["train", "val", "test"]:
            data['xtime_' + category] = build_time_embedding(data['xtime_' + category], data_dir[8:], time2id, args.period_time)
            data['ytime_' + category] = build_time_embedding(data['ytime_' + category], data_dir[8:], time2id, args.period_time)
            
            # data['xtime_' + category]  = build_real_time(data['xtime_' + category])
            # data['ytime_' + category] = build_real_time(data['ytime_' + category])
        if args.period_time:
            args.num_time = len(time2id)
        else:
            args.num_time = len(time2id) // 2
        

    # dataloader
    if if_time_embedding and args.od_flag:
        data["train_loader"] = Dataloader((data['x_train'], data['xtime_train'], data['xod_train'], data['xdo_train']), (data['y_train'], data['ytime_train']), 
                                batch_size, shuffle=True, if_time_embedding=True, od_flag=True)
        data["val_loader"] = Dataloader((data['x_val'], data['xtime_val'], data['xod_val'], data['xdo_val']), (data['y_val'], data['ytime_val']),
                                batch_size, shuffle=False, if_time_embedding=True, od_flag=True)
        data["test_loader"] = Dataloader((data['x_test'], data['xtime_test'], data['xod_test'], data['xdo_test']), (data['y_test'], data['ytime_test']),
                                batch_size, shuffle=False, if_time_embedding=True, od_flag=True)
    elif if_time_embedding:
        data["train_loader"] = Dataloader((data['x_train'], data['xtime_train']), (data['y_train'], data['ytime_train']), 
                                batch_size, shuffle=True, if_time_embedding=True, constra_time=args.constrative_time, data=args.data)
        data["val_loader"] = Dataloader((data['x_val'], data['xtime_val']), (data['y_val'], data['ytime_val']),
                                batch_size, shuffle=False, if_time_embedding=True)
        data["test_loader"] = Dataloader((data['x_test'], data['xtime_test']), (data['y_test'], data['ytime_test']),
                                batch_size, shuffle=False, if_time_embedding=True)
    else:
        data["train_loader"] = Dataloader(data['x_train'], data['y_train'], batch_size, shuffle=True)
        data["val_loader"] = Dataloader(data['x_val'], data['y_val'], batch_size, shuffle=False)
        data["test_loader"] = Dataloader(data['x_test'], data['y_test'], batch_size, shuffle=False)

    return data


class Dataloader():
    '''
    get the dataloader for the input of every iter.
    '''
    def __init__(self, X, Y, batch_size, shuffle=False, pad_with_last_sample=True, if_time_embedding=False, od_flag=False, constra_time=False, data="SH"):
        self.batch_size = batch_size
        self.if_time_embedding = if_time_embedding
        self.data = data
        self.od_flag = od_flag
        self.constra_time = constra_time
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(X) % batch_size)) % batch_size
            if if_time_embedding and od_flag:
                X, xtime, OD, DO = X
                Y, ytime = Y
                xtime_padding = np.repeat(xtime[-1:], num_padding, axis=0)
                ytime_padding = np.repeat(ytime[-1:], num_padding, axis=0)
                xtime = np.concatenate([xtime, xtime_padding], axis=0)
                ytime = np.concatenate([ytime, ytime_padding], axis=0)
                OD_padding = np.repeat(OD[-1:], num_padding, axis=0)
                DO_padding = np.repeat(DO[-1:], num_padding, axis=0)
                OD = np.concatenate([OD, OD_padding], axis=0)
                DO = np.concatenate([DO, DO_padding], axis=0)
            elif if_time_embedding:
                X, xtime = X
                Y, ytime = Y
                xtime_padding = np.repeat(xtime[-1:], num_padding, axis=0)
                ytime_padding = np.repeat(ytime[-1:], num_padding, axis=0)
                xtime = np.concatenate([xtime, xtime_padding], axis=0)
                ytime = np.concatenate([ytime, ytime_padding], axis=0)

            x_padding = np.repeat(X[-1:], num_padding, axis=0)
            y_padding = np.repeat(Y[-1:], num_padding, axis=0)
            X = np.concatenate([X, x_padding], axis=0)
            Y = np.concatenate([Y, y_padding], axis=0)

        self.size = len(X)
        self.num_batch = int(self.size // self.batch_size)
        self.X = X
        self.Y = Y
        
        if self.od_flag:
            self.od = OD
            self.do = DO

        if if_time_embedding:
            self.xtime = xtime
            self.ytime = ytime
    
    def shuffle(self):
        permutation = np.random.permutation(self.size)
        self.X, self.Y = self.X[permutation], self.Y[permutation]

        if self.if_time_embedding:
            self.xtime, self.ytime = self.xtime[permutation], self.ytime[permutation]
        
        if self.constra_time:
            # build constrative time learning sample

            self.cons_time = build_constrative_sample(self.xtime, self.ytime, self.batch_size, self.data)
            
        if self.od_flag:
            self.od, self.do = self.od[permutation], self.do[permutation]

    def gen_iterator(self):
        self.current_ind = 0
        def _wrapper_():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.X[start_ind: end_ind, ...]
                y_i = self.Y[start_ind: end_ind, ...]

                cuda = True if torch.cuda.is_available() else False
                TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
                TensorLong = torch.cuda.LongTensor if cuda else torch.LongTensor
                x_i, y_i = TensorFloat(x_i), TensorFloat(y_i)
                if self.if_time_embedding and self.od_flag:
                    xtime_i = self.xtime[start_ind: end_ind, ...]
                    ytime_i = self.ytime[start_ind: end_ind, ...]
                    xtime_i, ytime_i = TensorLong(xtime_i), TensorLong(ytime_i)
                    od_i = TensorFloat(self.od[start_ind: end_ind, ...])
                    do_i = TensorFloat(self.do[start_ind: end_ind, ...])
                    yield(x_i, xtime_i, y_i, ytime_i, od_i, do_i)
                elif self.if_time_embedding and self.constra_time:
                    xtime_i = self.xtime[start_ind: end_ind, ...]
                    ytime_i = self.ytime[start_ind: end_ind, ...]
                    xtime_i, ytime_i = TensorLong(xtime_i), TensorLong(ytime_i)
                    cons_time = TensorLong(self.cons_time[self.current_ind])
                    yield(x_i, xtime_i, y_i, ytime_i, cons_time)
                elif self.if_time_embedding:
                    xtime_i = self.xtime[start_ind: end_ind, ...]
                    ytime_i = self.ytime[start_ind: end_ind, ...]
                    xtime_i, ytime_i = TensorFloat(xtime_i), TensorFloat(ytime_i)
                    yield(x_i, xtime_i, y_i, ytime_i)
                else:
                    yield (x_i, y_i)

                self.current_ind += 1
        return _wrapper_()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


class Graph_loader(object):
    def __init__(self, num_stat, triples, n_neighbor=10):
        super(Graph_loader, self).__init__()
        self.num_stat = num_stat
        self.n_neighbor = n_neighbor
        self.triples = load_pickle(triples)

        self._consKG_()

    def _consKG_(self):
        self.kg = {}
        for e in self.triples:
            h, r, t = e[0], e[1], e[2]
            if h in self.kg:
                self.kg[h].append((r,t))
            else:
                self.kg[h] = [(r,t)]

    def _genA_(self):
        """
        generater adjacent matrix by sampling
        adj_ent[h_idx] = [t1, t2, t3, ....] resample, it's bad.
        """
        self.adj_ent = torch.empty(self.num_stat, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_stat, self.n_neighbor, dtype=torch.long)

        for e in self.kg:
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)
            
            self.adj_ent[e] = torch.LongTensor([t for _, t in neighbors])
            self.adj_rel[e] = torch.LongTensor([r for r, _ in neighbors])


def build_constrative_sample(xtime, ytime, batch_size, data):
    samples, samples_index = [], []

    for i in range(xtime.shape[0] // batch_size):
        start_ind = batch_size * i
        end_ind = min(xtime.shape[0], batch_size * (i + 1))
        x, y = xtime[start_ind:end_ind, ...], ytime[start_ind:end_ind, ...] # [B,4], [B,4]
        xy = np.concatenate((x,y),axis=1)
        
        # 根据编解码的长度，确定一个number值
        if data in ["taxi", "bike"]:
            number = 23 # 12+12-1
        else:
            number = 7
        # get anchor
        anchor_index = np.random.randint(low=0, high=number, size=(batch_size))
        anchor = xy[[j for j in range(batch_size)], anchor_index].reshape(-1,1) # [B, 1]

        # get adjacent dependens on anchor_idx
        if number == 7:
            adj = [-2, -1, 1, 2] # relative
        else:
            adj = [-2, -1, 1, 2]
        adjacent_index = np.random.choice(adj, (batch_size)) + anchor_index
        adjacent_index[adjacent_index < 0] += np.random.randint(low=1, high=2, size=1)[0]
        adjacent_index[adjacent_index > number] -= np.random.randint(low=1, high=2, size=1)[0]
        adjacent_index[adjacent_index < 0] = 0
        adjacent_index[adjacent_index > number] = number
        anchor_adj = xy[[j for j in range(batch_size)], adjacent_index].reshape(-1,1)

        # get distant dependes on anchor_idx
        if number == 7:
            dis = [-4, -3, 3, 4]
        else:
            dis = [-4, -3, 3, 4]
        # add or minus the large index value
        dist_index = np.random.choice(dis, (batch_size)) + anchor_index
        if number == 7:
            dist_index[dist_index < 0] += np.random.randint(low=5, high=7, size=1)[0] # 5 and 7, number
            dist_index[dist_index > number] -= np.random.randint(low=5, high=7, size=1)[0]
        else:
            dist_index[dist_index < 0] += np.random.randint(low=19, high=22, size=1)[0] # 5 and 7, number
            dist_index[dist_index > number] -= np.random.randint(low=19, high=22, size=1)[0]
        dist_index[dist_index < 0] = 0
        dist_index[dist_index > number] = number
        anchor_dis = xy[[j for j in range(batch_size)], dist_index].reshape(-1, 1)

        # get another batch sample
        _ = []
        for i in range(batch_size):
            _.append(np.random.choice(np.delete(xy, i, axis=0).flatten(), 1)[0])
        anchor_bat = np.array(_).reshape(batch_size, 1)

        samples.append(np.concatenate((anchor, anchor_adj, anchor_dis, anchor_bat), -1))
        # anchor_index = anchor_index.reshape(batch_size, 1)
        # adjacent_index = adjacent_index.reshape(batch_size, 1)
        # dist_index = dist_index.reshape(batch_size,1)   


    return np.array(samples)
