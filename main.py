"""
Main pipeline of DA-RNN.

@author Zhenye Na 05/21/2018
@modified 11/05/2019

References:
    [1] Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell.
        "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"
        arXiv preprint arXiv:1704.02971 (2017).
    [2] Chandler Zuo. "A PyTorch Example to Use RNN for Financial Prediction" (2017).
"""

import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable

from utils import *
from model import *

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="PyTorch implementation of paper 'A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction'")

    # Dataset setting
    parser.add_argument('--dataroot', type=str, default="./data/electricity.txt", help='path to dataset')
    parser.add_argument('--batchsize', type=int, default=512, help='input batch size [128]')

    # Encoder / Decoder parameters setting
    parser.add_argument('--nhidden_encoder', type=int, default=128, help='size of hidden states for the encoder m [64, 128]')
    parser.add_argument('--nhidden_decoder', type=int, default=128, help='size of hidden states for the decoder p [64, 128]')
    parser.add_argument('--ntimestep', type=int, default= 2 * 24 + 1, help='the number of time steps in the window T [10]')

    # Training parameters setting
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train [10, 200, 500]')
    parser.add_argument('--lr', type=list, default=[0.08], help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')
    parser.add_argument('--decay_rate', type=list, default=[0.8], help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')
    
    parser.add_argument('--data', type=str, required=False,default='./data/electricity.txt', help='location of the data file')
    parser.add_argument('--normalize', type=int, default=3)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--skipwindow', type=int, default= 24 * 4 * 30 - 1, help='skip window size')
    parser.add_argument('--window', type=int, default=2 * 24, help='window size')
    parser.add_argument('--index', type=int, default=320)
    parser.add_argument('--circle_length', type=int, default=24 * 4)
    parser.add_argument('--cuda', type=str, default=False)
    parser.add_argument('--save_1', type=str, default="./encoder_save/encoder.pt", help='path to save encoder')
    parser.add_argument('--save_2', type=str, default="./decoder_save/decoder_pt", help='path to save decoder')



    # parse the arguments
    args = parser.parse_args()

    return args


def main():
    """Main pipeline of DA-RNN."""
    print('\n' + '#'*20 + '\n')
    args = parse_args()

    # Read dataset
    print("==> Load dataset ...")
    Data = Data_utility(args.skipwindow,args.index, args.data, 0.7, 0.3, args.cuda, args.horizon, args.window, args.normalize);

    for epoch_lr in args.lr:
    	  
        for decay_rate in args.decay_rate:
        
          print('\n' + '#'*20 + '  learing_rate = ' + str(epoch_lr) + '  decay_rate = ' + str(decay_rate) + '  ' + '#'*20 + '\n')
          # Initialize model
          print("==> Initialize DA-RNN model ...")
          model = DA_RNN(
              Data,
              args.index,
              args.ntimestep,
        	  args.nhidden_encoder,
        	  args.nhidden_decoder,
        	  args.batchsize,
        	  epoch_lr,
        	  args.epochs,
              args,
        	  args.circle_length,
              decay_rate
    	  )

          # Train
          print("==> Start training ...")
          model.train()

          # Prediction
          y_pred = model.test()

          fig1 = plt.figure(figsize=(10,8))
          plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
          plt.savefig(('./pic_save/' + 'electricity_learing_rate=' + str(epoch_lr) + '_decay_rate=' + str(decay_rate) + '_nhidden_encoder_=_' + str(args.nhidden_encoder) + '_iter_1.png'))
          plt.close(fig1)

          fig2 = plt.figure(figsize=(10,8))
          plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
          plt.savefig(('./pic_save/' + 'electricity_learing_rate=' + str(epoch_lr) + '_decay_rate=' + str(decay_rate) + '_nhidden_encoder_=_' + str(args.nhidden_encoder) + '_iter_2.png'))
          plt.close(fig2)

          fig3 = plt.figure(figsize=(10,8))
          plt.plot(y_pred, label='Predicted')
          plt.plot(Data.dat[-y_pred.shape[0]:,args.index], label="True")
          plt.legend(loc='upper left')
          plt.savefig(('./pic_save/' + 'electricity_learing_rate=' + str(epoch_lr) + '_decay_rate=' + str(decay_rate) + '_nhidden_encoder_=_' + str(args.nhidden_encoder) + '_iter_3.png'))
          plt.close(fig3)
          print('Finished Training')

          fig4 = plt.figure(figsize=(10,8))
          plt.plot(y_pred[-1000:], label='Predicted')
          plt.plot(Data.dat[-1000:,args.index], label="True")
          plt.legend(loc='upper left')
          plt.savefig(('./pic_save/' + 'electricity_learing_rate=' + str(epoch_lr) + '_decay_rate=' + str(decay_rate) + '_nhidden_encoder_=_' + str(args.nhidden_encoder) + '_iter_4.png'))
          plt.close(fig4)
          print('Finished Training')

if __name__ == '__main__':
    main()

