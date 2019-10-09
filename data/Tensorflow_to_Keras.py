import os
import re
import numpy as np
import tensorflow as tf

import sys
sys.path.append('./utils/')
from resnet import *


tf_model_dir = '/Users/apple/Downloads/20180402-114759/'
npy_weights_dir = './model/keras/npy_weights/'
weights_dir = './model/keras/weights/'
model_dir = './model/keras/model/'

weights_filename = 'facenet_keras_weights.h5'
model_filename = 'facenet_keras.h5'

os.makedirs(npy_weights_dir,exist_ok=True)
os.makedirs(weights_dir,exist_ok=True)
os.makedirs(model_dir,exist_ok=True)

re_repeat = re.compile(r'Repeat_[0-9_]*b')
re_block8 = re.compile(r'Block8_[A-Za-z]')

def get_filename(key):
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('InceptionResnetV1_', '')

    # remove "Repeat" scope from filename
    filename = re_repeat.sub('B', filename)

    if re_block8.match(filename):
        # the last block8 has different name with the previous 5 occurrences
        filename = filename.replace('Block8', 'Block8_6')

    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'


def extract_tensors_from_checkpoint_file(filename, output_folder):
    reader = tf.train.load_checkpoint(filename)

    for key in reader.get_variable_to_shape_map():
        # not saving the following tensors
        if key == 'global_step':
            continue
        if 'AuxLogit' in key:
            continue

        # convert tensor name into the corresponding Keras layer weight name and save
        path = os.path.join(output_folder, get_filename(key))
        arr = reader.get_tensor(key)
        np.save(path, arr)

extract_tensors_from_checkpoint_file(tf_model_dir+'model-20180402-114759.ckpt-275', npy_weights_dir)
model = InceptionResNetV1()

print('Loading numpy weights from', npy_weights_dir)
for layer in model.layers:
    if layer.weights:
        weights = []
        for w in layer.weights:
            weight_name = os.path.basename(w.name).replace(':0', '')
            weight_file = layer.name + '_' + weight_name + '.npy'
            weight_arr = np.load(os.path.join(npy_weights_dir, weight_file))
            weights.append(weight_arr)
        layer.set_weights(weights)

print('Saving weights...')
model.save_weights(os.path.join(weights_dir, weights_filename))
print('Saving model...')
model.save(os.path.join(model_dir, model_filename))
