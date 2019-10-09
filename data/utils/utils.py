import os
import librosa
import scipy.io.wavfile as wavfile
import numpy as np


def mkdir(location):
    folder = os.path.exists(location)
    if not folder:
        os.mkdir(location)
        print("mkdir "+location+" ——success")
    else:
        print("location folder exists!!")


def download(loc,name,link,sr=16000,type='audio'): #download audio
    if type == 'audio':
        command = 'cd %s;' % loc
        command += 'youtube-dl -x --audio-format wav -o o' + name + '.wav ' + link + ';'
        command += 'ffmpeg -i o%s.wav -ar %d -ac 1 %s.wav;' % (name,sr,name)
        command += 'rm o%s.wav' % name
        os.system(command)


def cut(loc,name,start_time,end_time):
    length = end_time - start_time
    command = 'cd %s;' % loc
    command += 'sox %s.wav trim_%s.wav trim %s %s;' % (name,name,start_time,length)
    command += 'rm %s.wav' % name
    os.system(command)


def conc(loc,name,trim_clean=False):
    # concatenate the data in the loc (trim*.wav)
    command = 'cd %s;' % loc
    command += 'sox --combine concatenate trim_*.wav -o %s.wav;' % name
    if trim_clean:
        command += 'rm trim_*.wav;'
    os.system(command)


def mix(loc,name,file1,file2,start,end,trim_clean=False):
    command = 'cd %s;' % loc
    cut(loc,file1,start,end)
    cut(loc,file2,start,end)
    trim1 = '%s/trim_%s.wav' % (loc,file1)
    trim2 = '%s/trim_%s.wav' % (loc,file2)
    with open(trim1, 'rb') as f:
        wav1, wav1_sr = librosa.load(trim1, sr=None)  # time series data,sample rate
    with open(trim2, 'rb') as f:
        wav2, wav2_sr = librosa.load(trim2, sr=None)

    # compress the audio to same volume level
    wav1 = wav1 / np.max(wav1)
    wav2 = wav2 / np.max(wav2)
    assert wav1_sr == wav2_sr
    mix_wav = wav1*0.5+wav2*0.5

    path = '%s/%s.wav' % (loc,name)
    wavfile.write(path,wav1_sr,mix_wav)
    if trim_clean:
        command += 'rm trim_%s.wav;rm trim_%s.wav;' % (file1,file2)
    os.system(command)
