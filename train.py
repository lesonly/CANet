#!/usr/bin/python3
#coding=utf-8

import sys
import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data4 import dataset
from canet import ContextNet
import logging as logger
from lib4.data_prefetcher import DataPrefetcher
from lib4.lr_finder import LRFinder
import numpy as np
import matplotlib.pyplot as plt
from utils.generator import make_one_hot
#import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import argparse
import os
from utils.cutout import Cutout




# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '12345'


TAG = "ours"
SAVE_PATH = "ours"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="train_%s.log"%(TAG), filemode="w")


""" set lr """
def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
        annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps*ratio)
    last  = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur/total_steps)
    x = np.abs(cur*2.0/total_steps - 2.0*cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr)*cur + min_lr*first - base_lr*total_steps)/(first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1.-x)
        else:
            momentum = momentums[0]

    return lr, momentum


def box_loss(pred,mask, di,er):
    weit1  = 1+5*torch.abs(F.avg_pool2d(di, kernel_size=31, stride=1, padding=15)-di)
    weit2  = 1+5*torch.abs(F.avg_pool2d(er, kernel_size=31, stride=1, padding=15)-er)
    weight=weit1+weit2

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')

    wbce  = (weight*wbce).sum(dim=(2,3))/weight.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weight).sum(dim=(2,3))
    union = ((pred+mask)*weight).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

BASE_LR = 1e-3
MAX_LR = 0.1
FIND_LR = False #True#

# ------- 1. define loss function --------




def train(Dataset, Network):
    

    parser = argparse.ArgumentParser(description='PyTorch distributed training on cifar-10')
    parser.add_argument('--rank', default=0,
                        help='rank of current process')
    parser.add_argument('--word_size', default=4,
                        help="word size")
    parser.add_argument('--init_method', default='tcp://138.138.0.126:45678',
                        help="init-method")
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--batch_size','-b', default=8 ,type=int,help='batch-size')
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cfg    = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode='train', batch=args.batch_size, lr=0.05, momen=0.9, decay=5e-4, epoch=36)
    data   = Dataset.Data(cfg)
    #loader = DataLoader(data, batch_size=cfg.batch, shuffle=True,drop_last=True,num_workers=4)
    #train_sampler = torch.utils.data.distributed.DistributedSampler(data)
    #loader = DataLoader(data, batch_size=cfg.batch, num_workers=4, pin_memory=True, drop_last=True,sampler=train_sampler)
    loader = DataLoader(data, batch_size=cfg.batch, num_workers=4, pin_memory=True, drop_last=True,sampler=DistributedSampler(data,shuffle=True))

    prefetcher = DataPrefetcher(loader)

    ## network
    net    = Network(cfg)
    #net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
  
    net.train(True)
    #net=nn.DataParallel(net,device_ids=[0,1,2,3]) 
    # net.cuda()
    net.to(device)
    net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)      

   
    ## parameter
    base, edger,head = [], [],[]
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        elif 'edge_net' in name:
            edger.append(param) #, {'params':edger}
        else:
            head.append(param)
    optimizer   = torch.optim.SGD([{'params':base},{'params':edger},{'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)

   

    db_size = len(loader)
    if FIND_LR:
        os.environ['DISPLAY'] = '138.138.0.129:0.0'
        lr_finder = LRFinder(net, optimizer, criterion=None)
        lr_finder.range_test(loader,start_lr = 1e-6, end_lr=50, num_iter=330, step_mode="exp")
        plt.ion()
        lr_finder.plot()
        import pdb; pdb.set_trace()
    # #training


    global_step = 0
    p=None

    for epoch in range(cfg.epoch):
        loader.sampler.set_epoch(epoch)
        #scheduler.step()

        prefetcher = DataPrefetcher(loader)
        batch_idx = -1
        image, mask,thickedge_region,di,er= prefetcher.next()

        while image is not None:
            niter = epoch * db_size + batch_idx
            lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, cfg.epoch*db_size, niter, ratio=1.)
            optimizer.param_groups[0]['lr'] = 0.1 * lr #for backbone
            optimizer.param_groups[1]['lr'] = lr 
            optimizer.param_groups[2]['lr'] = lr #edger
            optimizer.momentum = momentum
        
            batch_idx += 1
            global_step += 1
             
            out2, out3, out4, out5,out5_c,out4_c= net(image)          
            loss2  = box_loss(out2,mask, di,er)
            loss3  = box_loss(out3,mask, di,er)
            loss4  = box_loss(out4,mask, di,er)
            loss5  = box_loss(out5,mask, di,er)    

  
            context_gt4 = make_one_hot(mask,0.0625)#18
            context_gt5 = make_one_hot(mask,0.03125)#9
            loss_context5 = F.binary_cross_entropy_with_logits(out5_c, context_gt5)
            loss_context4 = F.binary_cross_entropy_with_logits(out4_c, context_gt4)    

            loss   = loss2*1 + loss3*0.8 + (loss_context4+loss4)*0.6 + (loss_context5+loss5)*0.4

            optimizer.zero_grad()
    
            loss.backward()                          
     
            optimizer.step()
            
        
            if batch_idx % 10 == 0:               
                
                msg = '%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f | loss5=%.6f | loss=%.6f | loss_c4=%.6f | loss_c5=%.6f'% (
                    datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'],
                    loss.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(),loss.item(),loss_context4.item(),loss_context5.item())
                print(msg)
                logger.info(msg)
            
            image, mask,thickedge_region,di,er= prefetcher.next()
        
     
        if epoch==36:
           
            torch.save(net.module.state_dict(), cfg.savepath+'/CANet')#



if __name__=='__main__':
    
    train(dataset, ContextNet)
