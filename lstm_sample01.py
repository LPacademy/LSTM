from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import six
from numpy import random

a = 0.5
noiserate = a*0.05
pi = np.pi


def cycloid(t):
    x_t = a * (t-np.sin(t))
    y_t = a *(1-np.cos(t))
    return x_t, y_t

def make_noise(length):
    noize = noiserate*random.randn(length)
    return noize

def main():
    tmax = 6
    t = pi*np.linspace(0,tmax,100)
    x_t,y_t = cycloid(t)
    y_t_add_noise = y_t + make_noise(y_t.shape[0])
    plt.plot(x_t,y_t,'b')
    plt.plot(x_t,y_t_add_noise,'--r')
    plt.xlim(0,pi*tmax*a)
    plt.ylim(0,pi*tmax*a/2)
    plt.show()





if __name__ == '__main__':
    main()
