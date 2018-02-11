import os,sys,pdb
import numpy as np
import pandas as pd

sourcefile = 'training.csv'
datafile = 'train_data.csv'
labelfile ='train_label.csv'


Ys = []
Xs = []
with open(sourcefile,'rb') as f:
    f.readline()
    for line in f:
        line = line.strip()
        try:
            Y =  [ str( (np.float64(y)-48)/48.0 ) for y in line.split(',')[0:-1] ]
            if len(Y) != 30:
                print 'error Y'
                continue
            Y = ','.join(Y)
            X =  [ str( (np.int64(x) / 255.0) ) for x in line.split(',')[-1].split(' ')]
            if len(X) != 96 * 96:
                print 'error X'
                continue
            X = ','.join(X)
            
        except Exception,e:
            #print line.split(',')[0:-1]
            continue
        Ys.append(Y)
        Xs.append(X)

with open('train_data.csv', 'wb') as f:
    f.writelines('\r\n'.join(Xs))

with open('train_label.csv', 'wb') as f:
    f.writelines('\r\n'.join(Ys))
