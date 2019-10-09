import librosa
import os
import numpy as np
import scipy.io.wavfile as wavfile

audio_range = (0, 20)

if not os.path.exists('./norm_audio_train'):
    os.mkdir('./norm_audio_train')

for idx in range(audio_range[0], audio_range[1]):
    print('Processing audio %s'%idx)
    path = './audio_train/trim_audio_train%s.wav' % idx
    norm = './norm_audio_train/trim_audio_train%s.wav' % idx
    if os.path.exists(path):
        audio, _ = librosa.load(path, sr=16000)
        max = np.max(np.abs(audio))
        norm_audio = np.divide(audio, max)
        wavfile.write(norm,16000,norm_audio)


