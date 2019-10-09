from scipy.io import wavfile
import numpy as np


sample_rate, sig = wavfile.read('./norm_audio_train/trim_audio_train1.wav')

print("采样率: %d" % sample_rate)
print(sig)

if sig.dtype == np.int16:
    print("PCM16位整形")
if sig.dtype == np.float32:
    print("PCM32位浮点")