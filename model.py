""""
DA-RNN model architecture.

@author Zhenye Na 05/21/2018
@modified 11/05/2019

References:
    [1] Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell.
        "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"
        arXiv preprint arXiv:1704.02971 (2017).
    [2] Chandler Zuo. "A PyTorch Example to Use RNN for Financial Prediction" (2017).
"""

import matplotlib.pyplot as plt

import torch
import time
import numpy as np

from torch import nn
from torch import optim
from test_metrics import metric

from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self, T,
                 input_size,
                 encoder_num_hidden,
                 parallel=False):
        """Initialize an encoder in DA_RNN."""
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.parallel = parallel
        self.T = T

        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden,
            num_layers=1
        )

        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.T - 1,
            out_features=1
        )

    def forward(self, X):
        """forward.

        Args:
            X: input data

        """
        X_tilde = Variable(X.data.new(
            X.size(0), self.T - 1, self.input_size).zero_())
        X_encoded = Variable(X.data.new(
            X.size(0), self.T - 1, self.encoder_num_hidden).zero_())

        # Eq. 8, parameters not in nn.Linear but to be learnt
        # v_e = torch.nn.Parameter(data=torch.empty(
        #     self.input_size, self.T).uniform_(0, 1), requires_grad=True)
        # U_e = torch.nn.Parameter(data=torch.empty(
        #     self.T, self.T).uniform_(0, 1), requires_grad=True)

        # h_n, s_n: initial states with dimention hidden_size
        h_n = self._init_states(X)
        s_n = self._init_states(X)

        for t in range(self.T - 1):
            # batch_size * input_size * (2 * hidden_size + T - 1)
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)

            x = self.encoder_attn(
                x.view(-1, self.encoder_num_hidden * 2 + self.T - 1))

            # get weights by softmax
            alpha = F.softmax(x.view(-1, self.input_size), dim=1)

            # get new input for LSTM
            x_tilde = torch.mul(alpha, X[:, t, :])

            # Fix the warning about non-contiguous memory
            # https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.encoder_lstm.flatten_parameters()

            # encoder LSTM
            _, final_state = self.encoder_lstm(
                x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]

            X_tilde[:, t, :] = x_tilde
            X_encoded[:, t, :] = h_n

        return X_tilde, X_encoded

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder."""
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.encoder_num_hidden).zero_())


class Decoder(nn.Module):
    """decoder in DA_RNN."""

    def __init__(self, T, decoder_num_hidden, encoder_num_hidden):
        """Initialize a decoder in DA_RNN."""
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T

        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_num_hidden +
                      encoder_num_hidden, encoder_num_hidden),
            nn.Tanh(),
            nn.Linear(encoder_num_hidden, 1)
        )
        self.lstm_layer = nn.LSTM(
            input_size=1,
            hidden_size=decoder_num_hidden
        )
        self.fc = nn.Linear(encoder_num_hidden + 1, 1)
        self.fc_final = nn.Linear(decoder_num_hidden + encoder_num_hidden, 1)

        self.fc.weight.data.normal_()

        ################################## RNN_skip ###################################

        self.gru = nn.GRU(input_size = 1,hidden_size = 5,batch_first = True)
        self.linear1 = nn.Linear(in_features = 5,out_features = 1)
        self.linear2 = nn.Linear(in_features = 48,out_features = 1)
        
        ################################## RNN_skip ###################################


    def forward(self, X_encoded, y_prev, y_skip):
        """forward."""
        d_n = self._init_states(X_encoded)
        c_n = self._init_states(X_encoded)

        for t in range(self.T - 1):

            x = torch.cat((d_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           X_encoded), dim=2)

            beta = F.softmax(self.attn_layer(
                x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T - 1), dim=1)

            # Eqn. 14: compute context vector
            # batch_size * encoder_hidden_size
            context = torch.bmm(beta.unsqueeze(1), X_encoded)[:, 0, :]
            if t < self.T - 1:
                # Eqn. 15
                # batch_size * 1

                y_tilde = self.fc(
                    torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1))

                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, final_states = self.lstm_layer(
                    y_tilde.unsqueeze(0), (d_n, c_n))

                d_n = final_states[0]  # 1 * batch_size * decoder_num_hidden
                c_n = final_states[1]  # 1 * batch_size * decoder_num_hidden

        # Eqn. 22: final output
        y_pred = self.fc_final(torch.cat((d_n[0], context), dim=1))

        ################################## RNN_skip ###################################

        _,s_skip = self.gru(y_skip)
        s_skip = s_skip.squeeze(0)
        s_skip = self.linear1(s_skip)
        s_linear = self.linear2(y_prev)

#        y_pred = torch.cat((y_pred,s_skip),1);
#        y_pred = self.linear2(y_pred)

        ################################## RNN_skip ###################################

        return y_pred + s_skip + s_linear

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder."""
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.decoder_num_hidden).zero_())


class DA_RNN(nn.Module):
    """Dual-Stage Attention-Based Recurrent Neural Network."""

    def __init__(self, Data, index, T,
                 encoder_num_hidden,
                 decoder_num_hidden,
                 batch_size,
                 learning_rate,
                 epochs,
                 args,
                 circle_length,
                 decay_rate,
                 parallel=True):
        """initialization."""
        super(DA_RNN, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.parallel = parallel
        self.shuffle = False
        self.epochs = epochs
        self.T = T
        self.Data = Data
        self.args = args
        self.index = index
        self.circle_length = circle_length
        self.loss_fn = nn.MSELoss()
        self.decay_rate = decay_rate

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

        self.Encoder = Encoder(input_size=self.Data.train[0].shape[2],
                               encoder_num_hidden=encoder_num_hidden,
                               T=T).to(self.device)
        self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                               decoder_num_hidden=decoder_num_hidden,
                               T=T).to(self.device)

        # Loss function
        self.criterion = nn.MSELoss()

        if self.parallel:
            self.Encoder = nn.DataParallel(self.Encoder)
            self.Decoder = nn.DataParallel(self.Decoder)

        self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Encoder.parameters()),
                                            lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Decoder.parameters()),
                                            lr=self.learning_rate * 1.2)



    def train(self):

        """Training process."""
        iter_per_epoch = int(np.ceil(self.Data.train[0].shape[0] * 1. / self.batch_size))
        # 计算每个循环所需要的iteration数
        self.iter_losses = np.zeros(self.epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(self.epochs)

        n_iter = 0
        best_val = 10000000

        for epoch in range(self.epochs):
            # 选择是否对数据序列进行打乱
            begin = time.time()
            
            idx = 0
            for x, y_gt, y_prev, y_skip in self.Data.get_batches(self.Data.train[0],self.Data.train[1],self.Data.train[2],self.Data.train[3],self.batch_size,True):

                length = y_skip.shape[1] // self.circle_length
                y_skip = y_skip[:,: length * self.circle_length]
                y_skip = y_skip.view(-1,length,self.circle_length)
                y_skip = y_skip[:,:,[0]]


                loss = self.train_forward(x, y_prev, y_gt, y_skip)
                self.iter_losses[int(
                    epoch * iter_per_epoch + idx)] = loss

                idx += 1
                n_iter += self.batch_size

#                   for param_group in self.encoder_optimizer.param_groups:
#                        param_group['lr'] = param_group['lr'] * 0.9
#                    for param_group in self.decoder_optimizer.param_groups:
#                        param_group['lr'] = param_group['lr'] * 0.9

            self.epoch_losses[epoch] = np.mean(self.iter_losses[range(
                epoch * iter_per_epoch, (epoch + 1) * iter_per_epoch)])

            # if epoch % 10 == 0:
            end = time.time()
            print("Epochs: ", epoch, " Iterations: ", n_iter,
                    " Loss: ", self.epoch_losses[epoch]," run time statistics ",end - begin)
        
        
            if epoch % 10 == 0 and epoch != 0:
                for param_group in self.encoder_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * self.decay_rate
                for param_group in self.decoder_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * self.decay_rate

            if((epoch!=0) & (epoch % 4 == 0)) | ((epoch!=0) & (epoch % (self.epochs - 1) == 0)): 

                with torch.no_grad():
                    y_train_pred = self.test(on_train=True)
                    y_test_pred = self.test(on_train=False)
                
                # y_train_pred.shape = (28383,), y_test_pred.shape = (12168,)

                y_pred = np.concatenate((y_train_pred, y_test_pred))
                y_test_pred = y_test_pred.reshape(-1, 1)
                y_test_true = self.Data.dat[-y_test_pred.shape[0]:,[self.index]].reshape(-1,1)
            
                y_test_pred = y_test_pred.squeeze()
                y_test_true = y_test_true.squeeze()
            
                mae,mse,rmse,mape,mspe = metric(y_test_pred,y_test_true)
                print('test_score: mae = ',mae,' mse = ',mse,' rmse = ',rmse,' mape = ',mape,' mspe = ',mspe)


                if mse < best_val:
                    with open(self.args.save_1, 'wb') as f:
                        torch.save(self.Encoder, f)
                    with open(self.args.save_2, 'wb') as f:
                        torch.save(self.Decoder, f)
                    best_val = mse


            if(epoch != 0 ) & (epoch % (self.epochs - 1) == 0):

                # output_best model
                with open(self.args.save_1, 'rb') as f:
                    self.Encoder = torch.load(f)
                with open(self.args.save_2, 'rb') as f:
                    self.Decoder = torch.load(f)
                with torch.no_grad():
                    y_train_pred = self.test(on_train=True)
                    y_test_pred = self.test(on_train=False)

                y_pred = np.concatenate((y_train_pred, y_test_pred))
                y_test_pred = y_test_pred.reshape(-1, 1)
                y_test_true = self.Data.dat[-y_test_pred.shape[0]:,[self.index]].reshape(-1,1)

                y_test_pred = y_test_pred.squeeze()
                y_test_true = y_test_true.squeeze()

                mae,mse,rmse,mape,mspe = metric(y_test_pred,y_test_true)
                print('final_test: mae = ',mae,' mse = ',mse,' rmse = ',rmse,' mape = ',mape,' mspe = ',mspe)



    def train_forward(self, X, y_prev, y_gt, y_skip):


        """Forward pass."""
        # zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.Encoder(
            Variable(X.type(torch.FloatTensor).to(self.device)))

        y_pred = self.Decoder(input_encoded, Variable(
            y_prev.type(torch.FloatTensor).to(self.device)), Variable(y_skip.type(torch.FloatTensor).to(self.device)))

        # y_pred.shape = torch.Size([512, 1])

        y_true = Variable(
            y_gt.type(torch.FloatTensor).to(self.device))
        

        y_true = y_true.view(-1, 1)
        loss = self.criterion(y_pred, y_true)
        loss.backward()
        # y_pred.shape = (128，1)， y_true.shape= (128，1)

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()


        return loss.item()

    def test(self, on_train=False):
        """Prediction."""

        if on_train:
            y_pred = np.zeros(self.Data.train_end - self.Data.train_begin)
            data_0 = self.Data.train[0]
            data_1 = self.Data.train[1]
            data_2 = self.Data.train[2]
            data_3 = self.Data.train[3]
        else:
            y_pred = np.zeros(self.Data.valid[0].shape[0])
            data_0 = self.Data.valid[0]
            data_1 = self.Data.valid[1]
            data_2 = self.Data.valid[2]
            data_3 = self.Data.valid[3]
        # 如果测试训练集的话，则测前百分之七十；否则测后百分之三十

        i = 0
        for X, y_gt, y_prev, y_skip in self.Data.get_batches(data_0,data_1,data_2,data_3,self.batch_size,False):

            length = y_skip.shape[1] // self.circle_length
            y_skip = y_skip[:,: length * self.circle_length]
            y_skip = y_skip.view(-1,length,self.circle_length)
            y_skip = y_skip[:,:,[0]]

            # X.shape = (128,9,81) y_history.shape = (128,9)
            input_weighted, input_encoded = self.Encoder(
                        Variable(X.type(torch.FloatTensor).to(self.device)))

            y_pred[i:(i + X.shape[0])] = self.Decoder(input_encoded, Variable(
                y_prev.type(torch.FloatTensor).to(self.device)), Variable(y_skip.type(torch.FloatTensor).to(self.device))).cpu().data.numpy()[:, 0]

            i += X.shape[0]

        return y_pred

