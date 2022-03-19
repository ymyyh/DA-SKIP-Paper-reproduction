import torch
import random
import numpy as np;
from torch.autograd import Variable


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, skipwindow, index, file_name, train, valid, cuda, horizon, window, normalize = 2):
        self.skipwindow = skipwindow;
        self.index = index;
        self.cuda = cuda;
        self.P = window;
        self.h = horizon
        fin = open(file_name);
        self.rawdat = np.genfromtxt(fin,delimiter=',');
        # self.rawdat = np.loadtxt(fin,delimiter=',');
        self.dat = np.zeros(self.rawdat.shape);
        self.n, self.m = self.dat.shape;
        self.normalize = 2
        self.scale = np.ones(self.m);
        self._normalized(normalize);
        self._split(int(train * self.n), self.n);
        
        self.scale = torch.from_numpy(self.scale).float();
        tmp = self.valid[1] * self.scale.expand(self.valid[1].size(0), self.m);
            
        if self.cuda:
            self.scale = self.scale.cuda();
        self.scale = Variable(self.scale);
        
        self.rse = normal_std(tmp);
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)));
    
    def _normalized(self, normalize):
        #normalized by the maximum value of entire matrix.
       
        if (normalize == 0):
            self.dat = self.rawdat
            
        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat);
            
        #normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:,i]));
                self.dat[:,i] = self.rawdat[:,i] / np.max(np.abs(self.rawdat[:,i]));

        if (normalize == 3):
            split_pos = int(self.dat.shape[0] * 0.7) 
            self.dat = self.rawdat
            self.dat[:,self.index] = self.dat[:,self.index] - np.mean(self.dat[:split_pos,self.index]);
            
        
    def _split(self, train, valid):

        # random.seed(100)
        # total_list = list(range(self.P+self.h-1,self.n))
        # random.shuffle(total_list)
        
        train_set = range(self.skipwindow+self.h-1, train);
        valid_set = range(train, valid);

        self.train_begin = self.P+self.h-1
        self.train_end = train
        self.valid_end = valid
        # len(train_set) = 5048; len(valid_set) = 2277;       

        self.train = self._batchify(train_set, self.h);
        self.valid = self._batchify(valid_set, self.h);

        
        
    def _batchify(self, idx_set, horizon):
        
        n = len(idx_set);
        X = torch.zeros((n,self.P,self.m));
        Y = torch.zeros((n,1));
        Y_prev = torch.zeros((n,self.P,1))
        Y_skip = torch.zeros((n,self.skipwindow,1))

        
        for i in range(n):
            end = idx_set[i] - self.h + 1;
            start = end - self.P;
            skip_start = end - self.skipwindow;

            X[i,:,:] = torch.from_numpy(self.dat[start:end, :]);
            Y[i,:] = torch.from_numpy(self.dat[idx_set[i], [self.index]]);
            Y_prev[i,:,:] = torch.from_numpy(self.dat[start:end, [self.index]]);
            Y_skip[i,:,:] = torch.from_numpy(self.dat[skip_start:end,[self.index]])

        X = np.delete(X, self.index, axis=2)

        return [X, Y, Y_prev, Y_skip];

    def get_batches(self, inputs, targets, prev, skip, batch_size, shuffle=False):

        targets = targets.squeeze()
        prev = prev.squeeze()
        skip = skip.squeeze()

        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]; Y = targets[excerpt]; Y_prev = prev[excerpt]; Y_skip = skip[excerpt];
            if (self.cuda):
                X = X.cuda();
                Y = Y.cuda();
                Y_prev = Y_prev.cuda();
                Y_skip = Y_skip.cuda();  
            yield Variable(X), Variable(Y),Variable(Y_prev),Variable(Y_skip);
            start_idx += batch_size
