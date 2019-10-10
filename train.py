import sys

sys.path.append('./model/model/')
import AV_model as AV
from option import ModelMGPU, latest_file
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras.models import Model, load_model
from data_load import AVGenerator
from keras.callbacks import TensorBoard
from keras import optimizers
import os
from loss import audio_discriminate_loss2 as audio_loss
import tensorflow as tf

# Resume Model
resume_state = False

# Parameters
people_num = 2
epochs = 10
initial_epoch = 0
batch_size = 1
gamma_loss = 0.1
beta_loss = gamma_loss * 2

# Accelerate Training Process
workers = 8
MultiProcess = False
NUM_GPU = 0

# PATH
model_path = './saved_AV_models'  # model path
database_path = 'data/'

# create folder to save models
folder = os.path.exists(model_path)
if not folder:
    os.makedirs(model_path)
    print('create folder to save models')
filepath = model_path + "/AVmodel-" + str(people_num) + "p-{epoch:03d}-{val_loss:.5f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


# automatically change lr
def scheduler(epoch):
    ini_lr = 0.00001
    lr = ini_lr
    if epoch >= 5:
        lr = ini_lr / 5
    if epoch >= 10:
        lr = ini_lr / 10
    return lr


rlr = LearningRateScheduler(scheduler, verbose=1)
# format: mix.npy single.npy single.npy
trainfile = []
valfile = []
with open((database_path + 'AVdataset_train.txt'), 'r') as t:
    trainfile = t.readlines()
with open((database_path + 'AVdataset_val.txt'), 'r') as v:
    valfile = v.readlines()

# the training steps
if resume_state:
    latest_file = latest_file(model_path + '/')
    AV_model = load_model(latest_file, custom_objects={"tf": tf})
    info = latest_file.strip().split('-')
    initial_epoch = int(info[-2])
else:
    AV_model = AV.AV_model(people_num)

train_generator = AVGenerator(trainfile, database_path=database_path, batch_size=batch_size, shuffle=True)
val_generator = AVGenerator(valfile, database_path=database_path, batch_size=batch_size, shuffle=True)

if NUM_GPU > 1:
    parallel_model = ModelMGPU(AV_model, NUM_GPU)
    adam = optimizers.Adam()
    loss = audio_loss(gamma=gamma_loss, beta=beta_loss, people_num=people_num)
    parallel_model.compile(loss=loss, optimizer=adam)
    print(AV_model.summary())
    parallel_model.fit_generator(generator=train_generator,
                                 validation_data=val_generator,
                                 epochs=epochs,
                                 workers=workers,
                                 use_multiprocessing=MultiProcess,
                                 callbacks=[TensorBoard(log_dir='./log_AV'), checkpoint, rlr],
                                 initial_epoch=initial_epoch
                                 )
if NUM_GPU <= 1:
    adam = optimizers.Adam()
    loss = audio_loss(gamma=gamma_loss, beta=beta_loss, people_num=people_num)
    AV_model.compile(optimizer=adam, loss=loss)
    print(AV_model.summary())
    AV_model.fit_generator(generator=train_generator,
                           validation_data=val_generator,
                           epochs=epochs,
                           workers=workers,
                           use_multiprocessing=MultiProcess,
                           callbacks=[TensorBoard(log_dir='./log_AV'), checkpoint, rlr],
                           initial_epoch=initial_epoch
                           )
