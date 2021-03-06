# -*- coding : utf-8 -*- 
'''
This is a neural network model fot CNN.
The model is too simple to learn , so you can modify it easiliy with like bath norm and so on.
It consists of 3 layers that include a Convolutional layer and max pooling layer each.

'''
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

class CNN(chainer.Chain):
    def __init__(self,inpu_layer=3,output_n=3):
        super(CNN,self).__init__(
            conv1 = F.Convolution2D(inpu_layer,32,3,pad=1),
            conv2 = F.Convolution2D(32,32,3,pad=1),
            conv3 = F.Convolution2D(32,32,3,pad=1),
            conv4 = F.Convolution2D(32,32,3,pad=1),
            bnorm1=L.BatchNormalization(32),
            bnorm2=L.BatchNormalization(32),
            bnorm3=L.BatchNormalization(32),
            bnorm4=L.BatchNormalization(32),
            l1 = L.Linear(None,512),
            l2 = L.Linear(512,output_n),
        )

    def __call__(self,x,train=True):

        h1 = self.bnorm1(F.relu(self.conv1(x)))
        h2 = F.dropout(F.max_pooling_2d(h1, 2),train=train,ratio=0.2)
        h3 = self.bnorm2(F.relu(self.conv2(h2)))
        h4 = F.dropout(F.max_pooling_2d(h3, 2),train=train,ratio=0.2)
        h5 = self.bnorm3(F.relu(self.conv3(h4)))
        h6 = F.dropout(F.max_pooling_2d(h5, 2),train=train,ratio=0.2)
        h7 = F.relu(F.dropout(self.l1(h6),train=train,ratio=0.5))
        return self.l2(h7)
