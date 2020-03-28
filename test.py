import sys
sys.path.append ('./ model / model')
sys.path.append ('./ model / utils')
from keras.models import load_model
from option import ModelMGPU
import os
import scipy.io.wavfile as wavfile
import numpy as np
import utils
import tensorflow as tf



#parameters
people = 2
num_gpu=1

#path
model_path = './saved_AV_model/AVmodel-2p-099.h5'
result_path = './predict/'
os.makedirs(result_path,exist_ok=True)

database = './data/AV_model_database/mix/'
face_emb = './model/face_embedding/face1022_emb/'
print('Initialing Parameters......')

#loading data
print('Loading data ......')
test_file = []
with open('./data/AV_log/AVdataset_val.txt','r') as f:
    test_file = f.readlines()


def get_data_name(line,people=people,database=database,face_emb=face_emb):
    parts = line.split() # get each name of file for one testset
    mix_str = parts[0]
    name_list = mix_str.replace('.npy','')
    name_list = name_list.replace('mix-','',1)
    names = name_list.split('-')
    single_idxs = []
    for i in range(people):
        single_idxs.append(names[i])
    file_path = database + mix_str
    mix = np.load(file_path)
    face_embs = np.zeros((1,75,1,1792,people))
    for i in range(people):
        face_embs[1,:,:,:,i] = np.load(face_emb+"%05d_face_emb.npy"%single_idxs[i])

    return mix,single_idxs,face_embs

#result predict
av_model = load_model(model_path,custom_objects={'tf':tf})
if num_gpu>1:
    parallel = ModelMGPU(av_model,num_gpu)
    for line in test_file:
        mix,single_idxs,face_emb = get_data_name(line,people,database,face_emb)
        mix_ex = np.expand_dims(mix,axis=0)
        cRMs = parallel.predict([mix_ex,face_emb])
        cRMs = cRMs[0]
        prefix =''
        for idx in single_idxs:
            prefix +=idx+'-'
        for i in range(len(cRMs)):
            cRM =cRMs[:,:,:,i]
            assert cRM.shape ==(298,257,2)
            F = utils.fast_icRM(mix,cRM)
            T = utils.fase_istft(F,power=False)
            filename = result_path+str(single_idxs[i])+'.wav'
            wavfile.write(filename,16000,T)

if num_gpu<=1:
    for line in test_file:
        mix,single_idxs,face_emb = get_data_name(line,people,database,face_emb)
        mix_ex = np.expand_dims(mix,axis=0)
        cRMs = av_model.predict([mix_ex,face_emb])
        cRMs = cRMs[0]
        prefix =''
        for idx in single_idxs:
            prefix +=idx+'-'
        for i in range(len(cRMs)):
            cRM =cRMs[:,:,:,i]
            assert cRM.shape ==(298,257,2)
            F = utils.fast_icRM(mix,cRM)
            T = utils.fase_istft(F,power=False)
            filename = result_path+str(single_idxs[i])+'.wav'
            wavfile.write(filename,16000,T)
