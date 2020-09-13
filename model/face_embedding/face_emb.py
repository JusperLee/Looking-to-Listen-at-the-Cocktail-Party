import tensorflow as tf
import sys
sys.path.append('../lib/')
from tensorflow.python.framework import tensor_util
import numpy as np
from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

## paraeter
HDF5 = 1
MODEL_PATH = '../model/keras/facenet_keras.h5'
VALID_FRAME_LOG_PATH = '../../data/video_data/valid_face_text.txt'
FACE_INPUT_PATH = '../../data/video_data/face_input/'

data = np.random.randint(256,size=(1,160,160,3),dtype='int32')


###############
if HDF5:
    save_path = './face1022_emb/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = load_model(MODEL_PATH)
    model.summary()
    avgPool_layer_model = Model(inputs=model.input,outputs=model.get_layer('AvgPool').output)
    # print(avgPool_layer_model.predict(data))

    lines = []
    with open(VALID_FRAME_LOG_PATH, 'r') as f:
        lines = f.readlines()

    for line in lines:
        embtmp = np.zeros((75, 1, 1792))
        headname = line.strip()
        tailname = ''
        for i in range(1, 76):
            if i < 10:
                tailname = '_0{}.jpg'.format(i)
            else:
                tailname = '_' + str(i) + '.jpg'
            picname = headname + tailname
            # print(picname)
            I = mpimg.imread(FACE_INPUT_PATH + picname)
            I_np = np.array(I)
            I_np = I_np[np.newaxis, :, :, :]
            # print(I_np.shape)
            # print(avgPool_layer_model.predict(I_np).shape)
            embtmp[i - 1, :] = avgPool_layer_model.predict(I_np)

        # print(embtmp.shape)
        people_index = int(line.strip().split('_')[1])
        npname = '{:05d}_face_emb.npy'.format(people_index)
        print(npname)

        np.save(save_path + npname, embtmp)
        with open('faceemb_dataset.txt', 'a') as d:
            d.write(npname + '\n')






