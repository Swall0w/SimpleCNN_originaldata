# -*- coding : utf-8 -*- 
from model import CNN
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
#from chainer.datasets import tuple_dataset
import util

import numpy as np
import sys
import pickle
import numpy as np
#from skimage import io
#import glob
#from PIL import Image
import os

def train_model(EPOCH_NUM=100,BATCH_NUM=20):
    epoch_n = EPOCH_NUM
    batch_n = BATCH_NUM

    train, test = util.load_imageset()

# Initializing model
    model = L.Classifier(CNN(3))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

# Setting Iteration
    train_iter = chainer.iterators.SerialIterator(train,batch_n)
    test_iter = chainer.iterators.SerialIterator(test,batch_n,repeat=False,shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = training.Trainer(updater, (epoch_n,'epoch'), out="result_recog")

    trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch','main/loss','validation/main/loss','main/accuracy','validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()
    serializers.save_npz('recog.model',model) 
    print('end')
    
if __name__ == '__main__':
    train_model()
