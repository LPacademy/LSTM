from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import chainer
import chainer.functions as F
from chainer import cuda, Variable, FunctionSet, optimizers
plt.style.use('ggplot')
import chainer.links as L
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
import os
import h5py
import time
import six
#import chainer.computational_graph as c
from numpy import random
import time


####################################################################################
#    2016/12/01
#    Author:  Hiroyuki Miyoshi
#    predict cycloid using LSTM
####################################################################################



#==========Initialize=========#
a = 0.5
noiserate = a*0.05
pi = np.pi




class DBLSTM(chainer.Chain):
   def __init__(self, n_in, n_units, n_out):
      super(DBLSTM, self).__init__(
            fo_l1=L.LSTM(n_in,n_units),
            ba_l1=L.LSTM(n_in, n_units),
            fo_l2=L.Linear(n_units, n_out),
            ba_l2=L.Linear(n_units, n_out),
      )

   def __call__(self,x ,train=True):
      lensize = x.data.shape[0]
      fo_layer = [self.fo_l1(F.reshape(F.get_item(x,i),[1,-1])) for i in range(0,lensize)]
      ba_layer = [self.ba_l1(F.reshape(F.get_item(x,i),[1,-1])) for i in range(lensize-1,-1,-1)]
      fo_layer2 = [self.fo_l2(F.reshape(fo_layer[i],[1,-1])) for i in range(0,lensize)]
      ba_layer2 = [self.ba_l2(F.reshape(ba_layer[i],[1,-1])) for i in range(lensize-1,-1,-1)]


      #return [F.dropout(fo_layer2[i] + ba_layer2[i] ) for i in range(0,lensize)]
      return [fo_layer2[i] + ba_layer2[i]  for i in range(0,lensize)]


   def reset_state(self):
        self.fo_l1.reset_state()

   def forward(self,x, t, train=True):
      y_pre = self(x, train)
      y_pre = F.vstack(y_pre)
      loss = F.mean_squared_error(y_pre,t)

      return loss

   def predict(self,x, train=False):
       y_pre = self(x,train)
       y_pre = F.vstack(y_pre)

       return y_pre.data






def cycloid(t):
    x_t = a * (t-np.sin(t))
    y_t = a *(1-np.cos(t))
    return x_t, y_t

def make_noise(length):
    noize = noiserate*random.randn(length)
    return noize

def main():

    tmax = 6
    tnum = 100


    t = pi*np.linspace(0,tmax,tnum)
    x_t,y_t = cycloid(t)
    print(y_t.shape[0])
    y_t_add_noise = y_t + make_noise(y_t.shape[0])
    #plt.plot(x_t,y_t,'b')
    #plt.plot(x_t,y_t_add_noise,'--r')
    #plt.xlim(0,pi*tmax*a)
    #plt.ylim(0,pi*tmax*a/2)
    #plt.show()
    start = time.time()

    n_in = 1
    n_out = 1
    n_units = 256
    n_epoch = 10000

    truncated_num = 100

    model =DBLSTM(n_in,n_units,n_out)
    optimizer = optimizers.AdaGrad(lr=0.01)
    optimizer.setup(model)
    model.reset_state()

    for epoch in range(1,n_epoch+1):
        x = np.empty((0,1))
        y = np.empty((0,1))
        loss_seq = 0
        model.reset_state()
        #select_num = random.randint(0,tnum- truncated_num)
        #for i in range(select_num,select_num+truncated_num):
        for i in range(1,tnum+1):
            x = np.append(x,x_t[i-1])
            y = np.append(y,y_t[i-1])
            if(i == tnum):
                x = np.reshape(x.astype(np.float32),[-1,1])
                y = np.reshape(y.astype(np.float32),[-1,1])
                loss = model.forward(chainer.Variable(x),chainer.Variable(y),train=True)
                model.zerograds()
                loss.backward()
                optimizer.update()
                loss.unchain_backward()
                loss_seq = loss_seq + loss.data
                loss = chainer.Variable(np.zeros((), dtype = np.float32))
                x = np.empty((0,1))
                y = np.empty((0,1))
        print ('#epoch:',epoch,'#train:',loss_seq,'     #time',time.time()-start)


        if(epoch%200 == 0):

            y_prd = np.empty((0,1))
            x = np.empty((0,1))
            for i in range(1,tnum+1):
                x = np.append(x,x_t[i-1])
                if(i == tnum):
                    x = np.reshape(x.astype(np.float32),[-1,1])
                    prd = model.predict(chainer.Variable(x),train=False)
                    #print(prd.shape)
                    y_prd = np.append(y_prd,prd)
                    #model.reset_state()
                    x = np.empty((0,1))

            #print(x_t.shape)
            #print(y_prd.shape)
            plt.plot(x_t,y_prd,'--r')
            plt.plot(x_t,y_t,'b')
            plt.show()








if __name__ == '__main__':
    main()
