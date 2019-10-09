import numpy as np
import keras


class AudioGenerator(keras.utils.Sequence):
    def __init__(self, filename, database_path, Xdim=(298, 257, 2), ydim=(298, 257, 2, 2), batch_size=4, shuffle=True):
        'Initialization'
        self.filename = filename
        self.Xdim = Xdim
        self.ydim = ydim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.database_path = database_path

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filename) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list
        filename_temp = [self.filename[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(filename_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.filename))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, filename_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.Xdim))
        y = np.empty((self.batch_size, *self.ydim))

        # Generate data
        for i, ID in enumerate(filename_temp):
            info = ID.strip().split(' ')
            X[i,] = np.load(self.database_path+'AV_model_database/mix/' + info[0])

            for j in range(2):
                y[i, :, :, :, j] = np.load(self.database_path+'AV_model_database/crm/' + info[j + 1])

        assert y[:,:,:,0] != y[:,:,:,1]
        return X, y


class AVGenerator(keras.utils.Sequence):
    def __init__(self, filename, database_path, X1dim=(298, 257, 2), X2dim=(75, 1, 1792, 2), ydim=(298, 257, 2, 2), batch_size=4, shuffle=True):
        'Initialization'
        self.filename = filename
        self.X1dim = X1dim
        self.X2dim = X2dim
        self.ydim = ydim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.database_path = database_path

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filename) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list
        filename_temp = [self.filename[k] for k in indexes]

        # Generate data
        [X1, X2], y = self.__data_generation(filename_temp)

        return [X1, X2], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.filename))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, filename_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X1 = np.empty((self.batch_size, *self.X1dim))
        X2 = np.empty((self.batch_size, *self.X2dim))
        y = np.empty((self.batch_size, *self.ydim))

        # Generate data
        for i, ID in enumerate(filename_temp):
            info = ID.strip().split(' ')
            X1[i, ] = np.load(self.database_path+'AV_model_database/mix/' + info[0])
            for j in range(2):
                y[i, :, :, :, j] = np.load(self.database_path+'AV_model_database/crm/' + info[j + 1])
                X2[i, :, :, :, j] = np.load(self.database_path+'../model/face_embedding/face1022_emb/' + info[j + 3])

        return [X1, X2], y
