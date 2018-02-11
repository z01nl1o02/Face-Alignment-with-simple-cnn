import os,sys,pdb
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
import mxnet as mx
import random
import csv

train_label_file = 'data/train_label.csv'
train_data_file = 'data/train_data.csv'

class FACE_CSV_ITER(mx.io.DataIter):
    def __init__(self,batchSize):
        self.batchSize_ = batchSize
        self.provideData_ = [
                mx.io.DataDesc("data", (batchSize,1,96,96), np.float32)
                ]
        self.provideLabel_ = [
                mx.io.DataDesc("softmax_label",(batchSize,30),np.float32)
                ]
        self.train_fd_ = open(train_data_file,'rb')
        self.label_fd_ = open(train_label_file,'rb')
        self.train_csv_ = csv.reader(self.train_fd_, delimiter=',')
        self.label_csv_ = csv.reader(self.label_fd_,delimiter=',')
        return
    def __iter__(self):
        return self
    def reset(self):
        self.train_csv_.seek(0)
        self.label_csv_.seek(0)
        self.train_csv = csv.reader( self.train_csv_, delimiter=',')
        self.label_csv = csv.reader( self.label_csv_,delimiter=',')
        return
    def __next__(self):
        return self.next()
    @property
    def provide_data(self):
        return self.provideData_
    @property
    def provide_label(self):
        return self.provideLabel_
    def next(self):
        Xs = []
        Ys = []
        try:
            for k in range(self.batchSize_):
                X = [np.float(x) for x in next(self.train_csv_)]
                Y = [np.float(x) for x in next(self.label_csv_)]
                X = np.asarray( np.reshape(X,(1,96,96)))
                Xs.append(X)
                Ys.append(Y)
            Xs = [ mx.nd.array(Xs)  ]
            Ys = [ mx.nd.array(Ys) ]
            return mx.io.DataBatch(data = Xs, label = Ys)
        except StopIteration:
            data_read = len(Xs)
            if data_read > 0:
                self.next()
                pad = self.batchSize_ - data_read
                for k in range(pad):
                    X = [np.float(x) for x in next(self.train_csv_)]
                    Y = [np.float(x) for x in next(self.label_csv_)]
                    X = np.asarray( np.reshape(X,(1,96,96)))
                    Xs.append(X)
                    Ys.append(Y)
                Xs = [ mx.nd.array(Xs) ]
                Ys = [ mx.nd.array(Ys) ]
                return mx.io.DataBatch(data = Xs, label = Ys)
            else:
                raise StopIteration
        return


def test_face_csv_iter():
    csviter = FACE_CSV_ITER(10)
    print csviter.provide_data, ',', csviter.provide_label
    k = 0
    for batch in csviter:
        print k,',',str(batch)
        k += 1
    csviter.reset()
    return


def cnn(outputSize = 30):
    data = mx.sym.var('data')
    
    conv1 = mx.sym.Convolution(data = data,kernel = (3,3),num_filter = 32,name = 'conv1')
    act1 = mx.sym.Activation(data = conv1,act_type='relu',name = 'act1')
    pool1 = mx.sym.Pooling(data=act1,pool_type='max',kernel=(2,2),stride=(2,2),name = 'pool1')
    dp1 = mx.sym.Dropout(data=pool1,p=0.25,name = 'dp1')
    
    
    conv2 = mx.sym.Convolution(data = dp1,kernel = (3,3),num_filter = 64,name = 'conv2')
    act2 = mx.sym.Activation(data = conv2,act_type='relu',name='act2')
    pool2 = mx.sym.Pooling(data=act2,pool_type='max',kernel=(2,2),stride=(2,2),name='pool2')
    dp2 = mx.sym.Dropout(data=pool2,p=0.25,name='dp2')
    
    conv3 = mx.sym.Convolution(data = dp2,kernel = (3,3),num_filter = 128,name='conv3')
    act3 = mx.sym.Activation(data = conv3,act_type='relu',name='act3')
    pool3 = mx.sym.Pooling(data=act3,pool_type='max',kernel=(2,2),stride=(2,2),name='pool3')
    dp3 = mx.sym.Dropout(data=pool3,p=0.25,name='dp3')
    
    conv4 = mx.sym.Convolution(data = dp3,kernel = (3,3),num_filter = 256,name='conv4')
    act4 = mx.sym.Activation(data = conv4,act_type='relu',name='act4')
    pool4 = mx.sym.Pooling(data=act4,pool_type='max',kernel=(2,2),stride=(2,2),name='pool4')
    dp4 = mx.sym.Dropout(data=pool4,p=0.25,name='dp4') 
    
    flatten = mx.sym.flatten(data = dp4,name='flatten')
    
    fc1 = mx.sym.FullyConnected(data=flatten,num_hidden=512,name='fc1')
    act5 = mx.sym.Activation(data = fc1, act_type='relu',name='act5')
    
    fc2 = mx.sym.FullyConnected(data = act5, num_hidden = outputSize,name='fc2')
    symbol = mx.sym.LinearRegressionOutput(data = fc2,name='softmax')
    
    #mx.viz.plot_network(symbol).view()
    return mx.mod.Module(symbol = symbol,context = mx.cpu())
    
def get_train_iter(batchSize = 100,outputSize = 30):
    train = []
    test = []
    
    X = []
    Y = []
    with open(train_data_file,'rb') as f:
        for line in f:
            line = [ np.float64(x) for x in line.strip().split(',') ]
            X.append( line )
    with open(train_label_file,'rb') as f:
        for line in f:
            line = [ np.float64(x) for x in line.strip().split(',') ]
            Y.append( line )
    k = 0
    for x, y in zip(X,Y):
        if k % 5 == 0:
            test.append( (x,y) )
        else:
            train.append( (x,y) )
        k += 1
    random.shuffle( train )
    total = len(train)
    trainX = np.zeros( (total, 1, 96, 96), dtype=np.float32)
    trainY = np.zeros( (total, 30), dtype=np.float32)
    total = len(test)
    testX = np.zeros( (total, 1, 96, 96), dtype=np.float32)
    testY = np.zeros( (total, 30), dtype=np.float32)
    for k,xy in enumerate(train):
        x,y = xy
        trainX[k,0,:,:] = np.reshape(x,(96,96))
        trainY[k,:] = np.reshape(y,(30,))
    for k,xy in enumerate(test):
        x,y = xy
        testX[k,0,:,:] = np.reshape(x,(96,96))
        testY[k,:] = np.reshape(y,(30,))
    train_iter = mx.io.NDArrayIter(trainX,trainY,batchSize)
    test_iter = mx.io.NDArrayIter(testX,testY,batchSize)
    return (train_iter, test_iter)
    
if __name__ == "__main__":
    test_face_csv_iter()
    print 'ok....'
    batchSize = 10
    outputSize = 30
    train_iter,test_iter = get_train_iter(batchSize, outputSize)
    
    model = cnn(outputSize)
    
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head) #enable log out
    model.fit(train_iter, eval_data = test_iter, optimizer='sgd',
    optimizer_params={'learning_rate':0.1,'wd':0.00}, #weight decay
        eval_metric='mse',
        batch_end_callback=mx.callback.Speedometer(batchSize,100),
        epoch_end_callback=mx.callback.do_checkpoint(".\\"),
        num_epoch=2000)
            
 
