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
    def __init__(self,path_prefix,batchSize):
        self.batchSize_ = batchSize
        self.provideData_ = [
                mx.io.DataDesc("data", (batchSize,1,96,96), np.float32)
                ]
        self.provideLabel_ = [
                mx.io.DataDesc("softmax_label",(batchSize,30),np.float32)
                ]
        print str(self.provideData_)
        self.data_fd_ = open(path_prefix + "data.csv",'rb')
        self.label_fd_ = open(path_prefix + 'label.csv','rb')
        self.data_csv_ = csv.reader(self.data_fd_, delimiter=',')
        self.label_csv_ = csv.reader(self.label_fd_,delimiter=',')
        return
    def __iter__(self):
        return self
    def reset(self):
        self.data_fd_.seek(0)
        self.label_fd_.seek(0)
        self.data_csv = csv.reader( self.data_fd_, delimiter=',')
        self.label_csv = csv.reader( self.label_fd_,delimiter=',')
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
                X = [np.float(x) for x in next(self.data_csv_)]
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
                    X = [np.float(x) for x in next(self.data_csv_)]
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
    csviter = FACE_CSV_ITER('data/split_test_',10)
    print csviter.provide_data, ',', csviter.provide_label
    k = 0
    for batch in csviter:
        print k,',',str(batch)
        k += 1
    csviter.reset()
    return


def cnn(rootdir, modelname, epoch,batchSize = 1):
    load_model = mx.model.load_checkpoint( os.path.join(rootdir, modelname), epoch)
    sym, arg_params, aug_params = load_model
    all_layers = sym.get_internals()
    model = mx.mod.Module( symbol = all_layers['softmax_output'], label_names = None, context = [mx.cpu()])
    model.bind(for_training=False,data_shapes = [('data',(batchSize, 1, 96, 96))])
    model.set_params(arg_params, aug_params,allow_missing = True)
    return model
  

    
def get_train_iter(batchSize = 100,outputSize = 30):

    train_iter = FACE_CSV_ITER('data/split_train_',batchSize)
    test_iter = FACE_CSV_ITER('data/split_test_',batchSize)
    return (train_iter, test_iter)


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
    

def plot_sample(x, y0, y1,axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y0[0,0::2]*48+48, y0[0,1::2]*48+48, marker='x', s=10)
    axis.scatter(y1[0,0::2]*48+48, y1[0,1::2]*48+48, marker='o', s=10)
    return

def run():
    
    batchSize = 1
    outputSize = 30
    train_iter,test_iter = get_train_iter(batchSize, outputSize)
    
    model = cnn('./','model',30,batchSize)
   
    for batch in test_iter:
        model.forward(batch)
        data = batch.data[0].asnumpy()*255 
        label = np.reshape( batch.label[0][0].asnumpy(), (1,-1) )
        output = model.get_outputs()[0].asnumpy()
        plot_sample(data,label,output,plt)
        plt.show()
if __name__=="__main__":
    run()
