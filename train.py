import argparse
import os
from time import time
from pathlib import Path as p

import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from utils import (load_data,
                   train_AutoEncoder, train_DEKM,
                   AutoEncoder, DEKM)



def get_arg():
    arg = argparse.ArgumentParser()
    arg.add_argument('-bs', default=128, type=int, help='batch size')
    arg.add_argument('-pre_epoch', type=int, help='epochs for train Autoencoder')
    arg.add_argument('-epoch', type=int, help='epochs for train DEKM')
    arg.add_argument('-k', type=int, help='num of clusters')
    arg.add_argument('-take', type=float, default=1., help='the size of data will be used in training')
    arg.add_argument('-save_dir', default='weight', help='location where model will be saved')
    arg.add_argument('-seed', type=int, default=None, help='torch random seed')
    arg = arg.parse_args()
    return arg
    

def main():
    arg = get_arg()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(arg.save_dir):
        os.makedirs(arg.save_dir, exist_ok=True) 
    else:
        for path in p(arg.save_dir).glob('*.png'):
            path.unlink()
        
    if arg.seed is not None:
        torch.manual_seed(arg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    tr_ds, test_ds = load_data(arg.take)
    
    print('\ntrain num:', tr_ds[0].shape[0])
    print('test num:', test_ds[0].shape[0])
    
    # train AutoEncoder
    ae = AutoEncoder().to(device)    
    print(f'\nAE param: {ae.num_param():.2f} M')
    opt = Adam(ae.parameters())
    t0 = time()
    train_AutoEncoder(ae, opt, tr_ds[0], test_ds[0], arg.bs, device, arg.pre_epoch, arg.save_dir)
    t1 = time()
    
    # train DEKM
    print('\nload the best encoder and build DEKM ...')
    dekm = DEKM(torch.load(f'{arg.save_dir}/pretrain_AE.pt').encoder).to(device)  
    print(f'DEC param: {dekm.num_param():.2f} M') 
    opt = Adam(dekm.parameters())
    t2 = time()
    train_DEKM(dekm, opt, tr_ds, test_ds, arg.bs, arg.k, device, arg.epoch, arg.save_dir)
    t3 = time()
    
    print(f'\ntrain AE time: {t1 - t0:.2f} s')
    print(f'train DEKM time: {t3 - t2:.2f} s')



if __name__ == '__main__':
    main()
    