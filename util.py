# -*- coding : utf-8 -*- 
from chainer.datasets import tuple_dataset
import numpy as np
from skimage import io
import glob
from PIL import Image
import os

def show_data(data):
    test_data = data
    img1 = np.rollaxis(test_data,0,3)   # change image array to RBG
    io.imshow(img1)
    io.show()

def load_imageset(datapath='./data/'):
    def _check_file_num(datapath='./data/'):
        dirs=[]
        for item in os.listdir(datapath):
            if os.path.isdir(os.path.join(datapath,item)):
                dirs.append(item)
        return len(dirs)
    print(_check_file_num(datapath=datapath))

    pathsAndLabels = []
    for index in range(_check_file_num(datapath=datapath)):
        path = datapath + str(index) + '/'
        pathsAndLabels.append(np.asarray([path,index]))

    # shafle data set
    allData = []
    for pathAndLabel in pathsAndLabels:
        path = pathAndLabel[0]
        label = pathAndLabel[1]
        imagelist = glob.glob(path + '*')
        for imgName in imagelist:
            allData.append([imgName, label])
    allData = np.random.permutation(allData)

    imageData = []
    labelData = []
    for pathAndLabel in allData:
        img = Image.open(pathAndLabel[0])
# convert 3 chanels to r,g,b images
        r,g,b = img.split()
        rImgData = np.asarray(np.float32(r)/255.0)
        gImgData = np.asarray(np.float32(g)/255.0)
        bImgData = np.asarray(np.float32(b)/255.0)
        imgData = np.asarray([rImgData,gImgData,bImgData])
        imageData.append(imgData)
        labelData.append(np.int32(pathAndLabel[1]))

    threshold = np.int32(len(imageData)/5*4)
    train = tuple_dataset.TupleDataset(imageData[:threshold], labelData[:threshold])
    test = tuple_dataset.TupleDataset(imageData[threshold:], labelData[threshold:])
    return train, test
