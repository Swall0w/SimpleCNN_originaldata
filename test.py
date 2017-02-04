# -*- coding : utf-8 -*- 
import chainer
import chainer.links as L
import numpy as np
import sys
import numpy as np
from PIL import Image
from model import CNN

def estimate_img(img):
    def _convert_chanel_to_cnn(img):
        r,g,b = img.split()
        rImgData = np.asarray(np.float32(r)/255.0)
        gImgData = np.asarray(np.float32(g)/255.0)
        bImgData = np.asarray(np.float32(b)/255.0)
        imgData = np.asarray([rImgData,gImgData,bImgData])
        return imgData
    def _predict_img(model,img):
        xdata = np.array([_convert_chanel_to_cnn(img)])
        data = model.predictor(xdata)
        data = data.data.flatten()
        return data
    def softmax(x):
        ex = np.exp(x - np.max(x))
        return ex/ex.sum()

    model = L.Classifier(CNN(3))
    serializers.load_npz('recog.model',model) 
    data = _predict_img(model,img)
    return softmax(data)
    
if __name__ == '__main__':
    img = Image.open(sys.argv[1])
    print(estimate_img(img))
    

