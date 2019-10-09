import sys

sys.path.append('../../model/utils')
import os
import librosa
import numpy as np
import utils
import itertools
import time
import random
import math
import scipy.io.wavfile as wavfile

data_range = (0, 20)  # data usage to generate database
audio_norm_path = os.path.expanduser("./norm_audio_train")
database_path = '../AV_model_database'
frame_valid_path = '../video_data/valid_face_text.txt'
num_speakers = 2
max_generate_data = 50


# initial data dir
def init_dir(path=database_path):
    if not os.path.exists(path):
        os.mkdir(path)

    if not os.path.exists('%s/mix' % path):
        os.mkdir('%s/mix' % path)

    if not os.path.isdir('%s/single' % path):
        os.mkdir('%s/single' % path)

    if not os.path.isdir('%s/crm' % path):
        os.mkdir('%s/crm' % path)

    if not os.path.isdir('%s/mix_wav' % path):
        os.mkdir('%s/mix_wav' % path)


# Generate datasets dir list
def generate_data_list(data_r=data_range, audio_norm_pth=audio_norm_path, frame_valid=frame_valid_path):
    audio_path_list = []
    frame_set = set()
    with open(frame_valid, 'r') as f:
        frames = f.readlines()

    for idx in range(len(frames)):
        frame = frames[idx].replace('\n', '').replace('frame_', '')
        frame_set.add(int(frame))

    for idx in range(data_r[0], data_r[1]):
        print('\rchecking...%d' % int(frame), end='')
        path = audio_norm_pth + '/trim_audio_train%d.wav' % idx
        if os.path.exists(path) and (idx in frame_set):
            audio_path_list.append((idx, path))

    print('\nlength of the path list: ', len(audio_path_list))
    return audio_path_list


# audio generate stft data(numpy)
def audio_to_numpy(audio_path_list, data_path=database_path, fix_sr=16000):
    for idx, path in audio_path_list:
        print('\r aduio numpy generating... %d' % ((idx / len(audio_path_list)) * 100), end='')
        data, _ = librosa.load(path, sr=fix_sr)
        data = utils.fast_stft(data)
        name = 'single-%05d' % idx
        with open('%s/single_TF.txt' % data_path, 'a') as f:
            f.write('%s.npy' % name)
            f.write('\n')

        np.save(('%s/single/%s.npy' % (data_path, name)), data)
    print()


# Divided into n parts according to the number of speakers
def split_to_mix(audio_path_list, data_path=database_path, partition=2):
    length = len(audio_path_list)
    part_len = length // partition
    start = 0
    part_idx = 0
    split_list = []
    while ((start + part_len) < length):
        part = audio_path_list[start:(start + part_len)]
        split_list.append(part)
        with open('%s/single_TF_part%d.txt' % (data_path, part_idx), 'a') as f:
            for idx, _ in part:
                name = 'single-%05d' % idx
                f.write('%s.npy' % name)
                f.write('\n')

        start += part_len
        part_idx += 1
    return split_list


# Mix a single audio （numpy）
def single_mix(combo_idx, split_list, datapath):
    assert len(combo_idx) == len(split_list)
    mix_rate = 1.0 / float(len(split_list))
    wav_list = []
    prefix = 'mix'
    mid_name = ''
    for part_idx in range(len(split_list)):
        idx, path = split_list[part_idx][combo_idx[part_idx]]
        wav, _ = librosa.load(path, sr=16000)
        wav_list.append(wav)
        mid_name += '-%05d' % idx

    mix_wav = np.zeros_like(wav_list[0])
    for wav in wav_list:
        mix_wav += wav * mix_rate

    wav_name = prefix + mid_name + '.wav'
    wavfile.write('%s/mix_wav/%s' % (datapath, wav_name), 16000, mix_wav)

    F_mix = utils.fast_stft(mix_wav)
    name = prefix + mid_name + '.npy'
    store_path = '%s/mix/%s' % (datapath, name)

    np.save(store_path, F_mix)

    with open('%s/mix_log.txt' % datapath, 'a') as f:
        f.write(name)
        f.write('\n')


# Mix all the audio to get n2 audio
def all_mix(split_list, data_path=database_path, partition=2):
    assert len(split_list) == partition
    print('mixing data....')
    num_mix = 1
    num_mix_check = 1
    for part in split_list:
        num_mix *= len(part)

    print('number of mix data: ', num_mix)

    part_len = len(split_list[-1])
    idx_list = [i for i in range(part_len)]
    combo_idx_list = itertools.product(idx_list, repeat=partition)
    for combo_idx in combo_idx_list:
        num_mix_check += 1
        single_mix(combo_idx, split_list, data_path)
        print('\rnum of completed mixing audio : %d' % num_mix_check, end='')
    print()


# Single audio generation complex mask map
def single_crm(idx_str_list, mix_path, data_path):
    F_mix = np.load(mix_path)
    mix_name = 'mix'
    mid_name = ''
    dataset_line = ''
    for idx in idx_str_list:
        mid_name += '-%s' % idx
        mix_name += '-%s' % idx
    mix_name += '.npy'
    dataset_line += mix_name

    for idx in idx_str_list:
        single_name = 'single-%s.npy' % idx
        path = '%s/single/%s' % (data_path, single_name)
        F_single = np.load(path)
        cRM = utils.fast_cRM(F_single, F_mix)

        last_name = '-%s' % idx
        cRM_name = 'crm' + mid_name + last_name + '.npy'

        store_path = '%s/crm/%s' % (data_path, cRM_name)
        np.save(store_path, cRM)

        with open('%s/crm_log.txt' % data_path, 'a') as f:
            f.write(cRM_name)
            f.write('\n')

        dataset_line += (' ' + cRM_name)

    with open('%s/dataset.txt' % data_path, 'a') as f:
        f.write(dataset_line)
        f.write('\n')


# all audio generation complex mask map
def all_crm(mix_log_path, data_path=database_path):
    with open(mix_log_path, 'r') as f:
        mix_list = f.read().splitlines()

    for mix in mix_list:
        mix_path = '%s/mix/%s' % (data_path, mix)
        mix = mix.replace('.npy', '')
        mix = mix.replace('mix-', '')
        idx_str_lsit = mix.split('-')
        single_crm(idx_str_lsit, mix_path, data_path)


# Classify generated data into training sets and verification sets
def train_test_split(dataset_log_path, data_range=[0, 20], test_ratio=0.1, shuffle=True, database_repo=database_path):
    with open(dataset_log_path, 'r') as f:
        data_log = f.read().splitlines()

    if data_range[1] > len(data_log):
        data_range[1] = len(data_log) - 1
    samples = data_log[data_range[0]:data_range[1]]
    if shuffle:
        random.shuffle(samples)

    length = len(samples)
    mid = int(math.floor(test_ratio * length))
    test = samples[:mid]
    train = samples[mid:]

    with open('%s/dataset_train.txt' % database_repo, 'a') as f:
        for line in train:
            f.write(line)
            f.write('\n')

    with open('%s/dataset_val.txt' % database_repo, 'a') as f:
        for line in test:
            f.write(line)
            f.write('\n')


if __name__ == '__main__':
    init_dir()
    audio_path_list = generate_data_list()
    audio_to_numpy(audio_path_list)
    split_list = split_to_mix(audio_path_list, partition=num_speakers)
    all_mix(split_list, partition=num_speakers)

    mix_log_path = '%s/mix_log.txt' % database_path
    all_crm(mix_log_path)

    dataset_log_path = '%s/dataset.txt' % database_path
    train_test_split(dataset_log_path, data_range=[0, max_generate_data])
