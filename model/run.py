import os
import sys

# set path
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)

import platform
import torch
import numpy as np
import torch.nn as nn
import argparse
from datetime import datetime
from model.TGCRN import TGCRN
from model.Trainer import Trainer
from script.TrainInits import init_seed, str_to_bool
from script.dataloader import load_dataset
from script.TrainInits import print_model_parameters
from script.metrics import MAE_torch
from script.dataloader import Graph_loader


#--------------------Parameter Setting-------------------------------------#
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default="../data/SHMetro", type=str, help="HZMetro, SHMetro, taxi or bike")
args.add_argument('--data', default='SH', type=str, help="HZ, SH, taxi, bike")
args.add_argument('--mode', default='train', type=str)
args.add_argument('--device', default="cuda:3", type=str, help='indices of GPUs')
args.add_argument('--debug', default="False", type=str_to_bool)
args.add_argument('--model', default="AGCRN", type=str, help="GRU, KGRU, DKGRU")
args.add_argument('--cuda', default=True, type=bool)

# switch
args.add_argument('--node_mode', default="random", type=str, help="kgr, kgc, random")
args.add_argument('--graph_direction', default="symm", type=str, help="symm, asym")
args.add_argument('--constrative_time', default="False", type=str_to_bool, help="if enable constrative learning for time")
args.add_argument('--ct_factor', default=0.01, type=float, help='constrative loss factor')
args.add_argument('--period', default="True", type=str_to_bool, help="if enable period function")
args.add_argument('--period_factor', default=0.3, type=float, help='periodic function factor')
args.add_argument('--period_time', default="False", type=str_to_bool, help="if enable two embedding for time")
args.add_argument('--time_station', default="True", type=str_to_bool, help="if enable station represenataion with time")
args.add_argument('--Seq_Dec', default="True", type=str_to_bool, help="if ture use sequential decoder, false one output")
args.add_argument('--time_embedding', default="True", type=str_to_bool, help="if use timeslot embedding")
args.add_argument('--od_flag', default="False", type=str_to_bool, help="if use od matrix, dynamic with real od value")

args.add_argument('--time_dim', default=64, type=int, help="the dimension of time embedding")

# data
args.add_argument('--lag', default=4, type=int, help="SH, HZ:4, TAXI:12, bike:12")
args.add_argument('--horizon', default=4, type=int, help="the predicted time steps, SH, HZ:4, TAXI:12")
args.add_argument('--num_nodes', default=288, type=int, help='HZ:80, SH:288, bike:250, taxi:266')
args.add_argument('--tod', default="False", type=str_to_bool)
args.add_argument('--normalizer', default="std", type=str)
args.add_argument('--column_wise', default="False", type=str_to_bool)
args.add_argument('--default_graph', default="True", type=str_to_bool)

#model
args.add_argument('--input_dim', default=2, type=int)
args.add_argument('--output_dim', default=2, type=int)
args.add_argument('--embed_dim', default=64, type=int)
args.add_argument('--rnn_units', default=64, type=int)
args.add_argument('--num_layers', default=2, type=int)
args.add_argument('--cheb_k', default=2, type=int)


#train
args.add_argument('--loss_func', default="mae", type=str)
args.add_argument('--seed', default=10, type=int)
args.add_argument('--batch_size', default=16, type=int)
args.add_argument('--epochs', default=1000, type=int)
args.add_argument('--lr_init', default=0.001, type=float)
args.add_argument('--lr_decay', default="True", type=str_to_bool)
args.add_argument('--lr_decay_rate', default=0.3, type=float)
args.add_argument('--lr_decay_step', default="5, 20, 40,70,90,100,130,150", type=str)
args.add_argument('--early_stop', default="True", type=str_to_bool)
args.add_argument('--early_stop_patience', default=15, type=int)
args.add_argument('--grad_norm', default="False", type=str_to_bool)
args.add_argument('--max_grad_norm', default=5, type=int)
args.add_argument('--teacher_forcing', default="False", type=str_to_bool)
args.add_argument('--real_value', default="True", type=str_to_bool, help = 'use real value for loss calculation')

#test
args.add_argument('--mae_thresh', default="False", type=str_to_bool)
args.add_argument('--mape_thresh', default=0., type=float)

#log
args.add_argument('--log_step', default=20, type=int)
args.add_argument('--plot', default="False", type=str_to_bool)
#-------------------------------------------------------------------------#

def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

#-------------------Initialize Training Device----------------------------#
args = args.parse_args()
init_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'


def main():
    #load dataset
    data = load_dataset(args.dataset, args.normalizer, args.batch_size, args, args.time_embedding)
    model = TGCRN(args)
    model = model.to(args.device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    print_model_parameters(model, only_num=False)
    
    if args.loss_func == 'mask_mae':
        loss = masked_mae_loss(data["scaler"], mask_value=0.0)
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(args.device)
    else:
        raise ValueError

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0.0001, amsgrad=False)
    
    lr_scheduler = None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_decay_steps,
                                                            gamma=args.lr_decay_rate)
    # log.....
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(current_dir,'experiments', args.dataset, current_time)
    args.log_dir = log_dir

    #start training or testing
    true_val_len = data["y_val"].shape[0]
    true_test_len = data["y_test"].shape[0]
    args.val_len = true_val_len
    args.test_len = true_test_len
    # graph_loader = Graph_loader(args.num_nodes, args.triples, args.n_neighbor)
    graph_loader = None

    trainer = Trainer(model, loss, optimizer, data["train_loader"], data["val_loader"], 
                    data["test_loader"], data["scaler"], args, graph_loader, lr_scheduler=lr_scheduler, ct_factor=args.ct_factor)
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        model.load_state_dict(torch.load('./data/HZMetro/20230116031928/best_model.pth'))
        print("Load saved model")
        trainer.test(model, trainer.args, data["test_loader"], data["scaler"], trainer.logger)
    else:
        raise ValueError

if __name__=="__main__":
    main()
