import numpy as np
from scipy.io import wavfile
import wave
import array
import os

def read_wav(wave_path):
    """
    这个文件用于读取wav文件，原始数据保存格式为wav文件，每个文件包含5个相同单词的书写信号。
    Parameters:
    ----------
        wave_path:读取路径，为文件的原始路径

    Returns:
    -------
        wave_data:读取的原始信号

    >>> audio_signal = read_wav(wave_path="DATA/source_signal.wav")
    """
    wave_read = wave.open(wave_path, "rb")
    params = wave_read.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = wave_read.readframes(nframes)
    wave_read.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data.shape = -1, nchannels
    wave_data = wave_data.T
    wave_data = wave_data[0]

    return wave_data

def read_pcm(pcm_path):  
    """
    该文件用于读取PCM格式文件，直接以流的方式进行读取。

    Parameters:
    ----------
        pcm_path:文件读取路径
    Returns:
    -------
        pcm_data:原始数据
    
    >>> audio_signal = read_pcm(pcm_path="DATA/source_signal.pcm")
    """
    pcm_read = open(pcm_path, 'rb')
    shortArray = array.array('h')
    size = int(os.path.getsize(pcm_path) / shortArray.itemsize)

    shortArray.fromfile(pcm_read, size)
    pcm_read.close()

    pcm_data = np.array(shortArray)
    return pcm_data

def read_wavs(file_path):
    """
    从路径file_path，读取并返回wav数据

    file_path: wav文件路径

    return: wav_data
    """
    wave_read = wave.open(file_path, "rb")
    params = wave_read.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = wave_read.readframes(nframes)
    wave_read.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data.shape = -1, nchannels
    wave_data = wave_data.T
    return wave_data / 32767, framerate