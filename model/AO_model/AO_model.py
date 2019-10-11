from keras import optimizers
from keras.layers import Input, Dense, Convolution2D, Bidirectional,TimeDistributed
from keras.layers import Flatten, BatchNormalization, Reshape
from keras.layers.core import Activation
from keras.models import Model, load_model
from keras.layers.recurrent import LSTM
from keras.initializers import he_normal,glorot_uniform



def AO_model(people_num=2):
    model_input = Input(shape=(298, 257, 2))
    print('0:', model_input.shape)

    conv1 = Convolution2D(96, kernel_size=(1, 7), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv1')(model_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    print('1:', conv1.shape)

    conv2 = Convolution2D(96, kernel_size=(7, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv2')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    print('2:', conv2.shape)

    conv3 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv3')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    print('3:', conv3.shape)

    conv4 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(2, 1), name='conv4')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    print('4:', conv4.shape)

    conv5 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(4, 1), name='conv5')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    print('5:', conv5.shape)

    conv6 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(8, 1), name='conv6')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    print('6:', conv6.shape)

    conv7 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(16, 1), name='conv7')(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    print('7:', conv7.shape)

    conv8 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(32, 1), name='conv8')(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    print('8:', conv8.shape)

    conv9 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv9')(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    print('9:', conv9.shape)

    conv10 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(2, 2), name='conv10')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)
    print('10:', conv10.shape)

    conv11 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(4, 4), name='conv11')(conv10)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)
    print('11:', conv11.shape)

    conv12 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(8, 8), name='conv12')(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)
    print('12:', conv12.shape)

    conv13 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(16, 16), name='conv13')(conv12)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)
    print('13:', conv13.shape)

    conv14 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(32, 32), name='conv14')(conv13)
    conv14 = BatchNormalization()(conv14)
    conv14 = Activation('relu')(conv14)
    print('14:', conv14.shape)

    conv15 = Convolution2D(8, kernel_size=(1, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv15')(conv14)
    conv15 = BatchNormalization()(conv15)
    conv15 = Activation('relu')(conv15)
    print('15:', conv15.shape)

    AVfusion = TimeDistributed(Flatten())(conv15)
    print('AVfusion:', AVfusion.shape)

    lstm = Bidirectional(LSTM(400,input_shape=(298,8*257),return_sequences=True),merge_mode='sum')(AVfusion)
    print('lstm:', lstm.shape)

    fc1 = Dense(600, name="fc1", activation='relu', kernel_initializer=he_normal(seed=27))(lstm)
    print('fc1:', fc1.shape)
    fc2 = Dense(600, name="fc2", activation='relu', kernel_initializer=he_normal(seed=42))(fc1)
    print('fc2:', fc2.shape)
    fc3 = Dense(600, name="fc3", activation='relu', kernel_initializer=he_normal(seed=65))(fc2)
    print('fc3:', fc3.shape)

    complex_mask = Dense(257 * 2 * people_num, name="complex_mask", kernel_initializer=glorot_uniform(seed=87))(fc3)
    print('complex_mask:', complex_mask.shape)

    complex_mask_out = Reshape((298, 257, 2, people_num))(complex_mask)
    print('complex_mask_out:', complex_mask_out.shape)

    # --------------------------- AO end ---------------------------
    AO_model = Model(inputs=model_input, outputs=complex_mask_out)


    return AO_model

