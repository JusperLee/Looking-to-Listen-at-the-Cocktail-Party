import sys
sys.path.append('../utils')
import utils
import pandas as pd


def make_audio(location, name, d_csv, start_idx, end_idx):
    for i in range(start_idx,end_idx):
        f_name = name + str(i)
        link = "https://www.youtube.com/watch?v="+d_csv.loc[i][0]
        start_time = d_csv.loc[i][1]
        end_time = start_time+3.0
        utils.download(location,f_name,link)
        utils.cut(location,f_name,start_time,end_time)
        print("\r Process audio... ".format(i) + str(i), end="")
    print("\r Finish !!", end="")



cat_train = pd.read_csv('../csv/avspeech_train.csv',header=None)
#cat_test = pd.read_csv('catalog/avspeech_test.csv')

# create 80000-90000 audios data from 290K
utils.mkdir('audio_train')
make_audio('audio_train','audio_train',cat_train,0,20)



