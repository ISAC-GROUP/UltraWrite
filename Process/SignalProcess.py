import scipy.signal as scisig
from scipy.fftpack import fft, fftshift
from sklearn import preprocessing
import numpy as np
import math
import matplotlib.pyplot as plt

import os
from . import DataIO, Display
# import DataIO, Display
import warnings

warnings.filterwarnings("ignore")

def audio2spetrum(audio_signal, window_size, overlap, FS, log):
    """
    这是一个将音频数据转化为频谱图的函数。

    Parameters
    ----------
        audio_signal:音频信号
        window_size:短时傅里叶变换窗口大小
        overlap:短时傅里叶变换重叠大小
        FS:采样率
        log:是否使用log的标志 [1/0]

    Returns:
        f:采样频率数组
        t:细分时间数组
        Sxx:频谱图
    """
    [f, t, Sxx] = scisig.spectrogram(audio_signal, FS, window="hamming", nperseg=window_size, noverlap=overlap, detrend=False)
    Sxx_magnitude = np.abs(Sxx)
    if log is True:
        PP = 10 * np.log10(Sxx_magnitude + 2.2204e-16)
    else:
        PP = Sxx_magnitude
    return f, t, PP

def transfer_data(root_path, save_path):
    """
    该函数实现将原始文件读取，并转存。转存时，将信息保留在文件名上。

    Parameters
    ----------
        root_path:原始路径
        save_path:转存路径
    
    Returns
    -------
        Source_filenames:原始文件名
        New_filenames:转存文件名
    
    >>> Source_filename, New_filenames = transfer_data(root_path="../../Acoudata/WDY", save_path="../DATA")
    """
    # 若目录不存在，则创建目录，否则不做操作
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
        print("dirs create successful!")
    else:
        print("dirs is exist!")
    
    # 10个书写的单词，将会成为文件名的组成部分
    Words_list = ["ME", "YOU", "HE", "AM", "ARE", "IS", "THE", "MAN", "HELLO", "WORLD"]
    Source_filenames = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            Source_filenames.append(root + "/" + file)
    
    New_filenames = []
    index = 0
    for filename in Source_filenames:
        data = DataIO.read_wav(wave_path=filename)
        user_name = filename.split("/")[-2]
        new_filename = save_path+ "/" + user_name + "_" + Words_list[int(index / 4)] + "_" + str((index % 4) + 1)+".npy"
        index += 1
        New_filenames.append(new_filename)
        np.save(new_filename, data)

    print("data restore successful!")
    return Source_filenames, New_filenames

def compare2file(file1, file2):
    """
    比较两个文件的数据是否完全一样
    在该系统中,文件只有可能是三种类型数据: 1-wav; 2-pcm; 3-npy;
    若print为True,则表示两个文件的数据完全一样,否则不是

    Parameters
    ----------
        file1:the file name of the first file 
        file2:the file name of the second file
    
    Returns:
        None
    
    >>> compare2file(file1 = "Source_filenames[0]", file2 = "New_filenames[0]")
    """

    if file1.endswith("wav"):
        data1 = DataIO.read_wav(wave_path=file1)
    elif file1.endswith("pcm"):
        data1 = DataIO.read_pcm(pcm_path=file1)
    else:
        data1 = np.load(file1)
    
    if file2.endswith("wav"):
        data2 = DataIO.read_wav(wave_path=file2)
    elif file2.endswith("pcm"):
        data2 = DataIO.read_pcm(pcm_path=file2)
    else:
        data2 = np.load(file2)

    print((data1 == data2).all())

def load_data(data_path):
    """
    根据路径，读取该路径下的所有npy文件
    文件名含有信息：用户名 _ label _ fileID

    在读取文件数据之前，会先将该文件的文件名上的信息进行读取保存

    Parameters
    ----------
        data_path:数据所在的目录

    Returns
    -------
        Users:用户id
        Labels:该文件对应的数据的标签
        Datas:文件内的数据
        Filenames:文件名

    >>> users, labels, datas, filenames = load_data(../DATA) 
    """
    Users = []
    Labels = []
    Datas = []
    Filenames = []

    for root, dirs, files in os.walk(data_path):
        for file in files:            
            flist = file.split("_")

            Users.append(flist[0])
            Labels.append(flist[1])

            filename = root + "/" + file
            Filenames.append(filename)
            Datas.append(np.load(filename))

    return Users, Labels, Datas, Filenames

def restore_and_check():
    """数据转存，并验证是否正确转存"""
    # root_path = "../../DATA/Acoudata/ZPZ"
    root_path = "../../DATA/temp"
    save_path = "../../DATA/tempnpy"
    Source_filenames, New_filenames = transfer_data(root_path, save_path)
    for i in range(len(Source_filenames)):
        file1 = Source_filenames[i]
        file2 = New_filenames[i]
        compare2file(file1, file2)

def filter(source_signal, order, cutoffs, FS, btype):
    """
    这是一个滤波函数，使用巴特沃斯滤波器，可以完成低通滤波、高通滤波、带通滤波、带阻滤波功能。

    Parameters
    ----------
        source_signal:原始信号
        order:滤波器阶数
        cutoffs:截止频段
        FS:采样率
        btype:滤波类型【lowpass, highpass, bandpass, bandstop】
    
    Returns
    -------
        signal:滤波后信号
    
    >>> signal = filter(source_signal, 8, [0,5000], 44100, "lowpass")
        该表达是为对 source_signal 进行一个 [0 : 5000] 的低通滤波
        滤波器阶数为 8
    """
    cutoffs = np.array(cutoffs)
    Wn = cutoffs / (FS / 2)
    b, a = scisig.butter(N=order, Wn=Wn, btype=btype)

    shape = source_signal.shape
    if shape[0] != 1:
        source_signal = source_signal.T

    signal = scisig.filtfilt(b, a, source_signal)
    signal = signal.T

    return signal

def matrix_scaler(matrix):
    """
    对矩阵进行归一化，使用scipy库集成的MinMaxScaler归一化函数，将数据归一化到（-1,1）之间
    
    在计算过程中，会先将矩阵压缩为 1D 向量，在归一化之后，在重置为原始shape

    Parameters
    ----------
        matrix:原始数据
    
    Returns
    -------
        result_matrix:归一化之后的数据
    
    >>> matrix_norm = matrix_scaler(matrix)
    """
    shape = matrix.shape

    temp_matrix = matrix.reshape(-1)
    minmaxscaler = preprocessing.MinMaxScaler()
    temp_matrix = minmaxscaler.fit_transform(temp_matrix.reshape(-1, 1)).reshape(-1)

    result_matrix = temp_matrix.reshape(shape)

    return result_matrix

def filter_noise(matrix, threshold):
    shape = matrix.shape
    result_matrix = np.zeros_like(matrix, dtype=np.float)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if matrix[i][j] < threshold:
                result_matrix[i][j] = 0
            else:
                result_matrix[i][j] = matrix[i][j]
    
    return result_matrix

def create_gaussian_kernel_2D(kernel_size=5, sigma=1):
    """
    生成 2D 高斯核
    
    高斯核值 kernel(x, y) = 1.0 / (2 * np.pi * sigma ** 2) * np.exp(-1.0 / (2 * sigma ** 2) * (x ** 2 + y ** 2))
    
    其中 radius是高斯核的中心

    Parameters
    ----------
        kernel_size:高斯核的尺寸，默认为5
        sigma:高斯分布的方差，默认为1
    
    Returns
    -------
        kernel:返回的高斯核
    """
    kernel = np.zeros(shape=(kernel_size, kernel_size), dtype=np.float)
    radius = kernel_size//2

    for y in range(-radius, radius + 1):  # [-r, r]
        for x in range(-radius, radius + 1):
            # 二维高斯函数
            v = 1.0 / (2 * np.pi * sigma ** 2) * np.exp(-1.0 / (2 * sigma ** 2) * (x ** 2 + y ** 2))
            kernel[y + radius, x + radius] = v  # 高斯函数的x和y值 vs 高斯核的下标值

    return kernel

def Gaussian_smoothing_2D(source_signal, ksize=5, sigma=1):
    """
    2维高斯平滑

    先调用create_gaussian_kernel_2D函数生成高斯核，然后再使用卷积的方案，进行平滑

    Parameters
    ----------
        source_signal:原始信号
        ksize:高斯核的尺寸
        sigma:高斯分布的方差
    
    Returns
    -------
        outputs:平滑后的数据
    """
    kernel = create_gaussian_kernel_2D(ksize, sigma)
    h, w = source_signal.shape
    h1, w1 = kernel.shape
    h_pad = (h1 - 1) // 2
    w_pad = (w1 - 1) // 2

    # 根据高斯核尺寸对原始数据先进性padding
    source_signal = np.pad(source_signal, pad_width=[(h_pad, h_pad), (w_pad, w_pad)], mode="constant", constant_values=0)
    outputs = np.zeros(shape=(h, w))
    for i in range(h):  # 行号
        for j in range(w):  # 列号
            outputs[i, j] = np.sum(np.multiply(source_signal[i: i + h1, j: j + w1], kernel))
    return outputs

def create_gaussian_kernal_1D(kernel_size, sigma):
    """
    生成 1D 高斯核
    
    高斯核值 kernel(x) = 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x/sigma) ** 2))
    
    其中 radius是高斯核的中心

    Parameters
    ----------
        kernel_size:高斯核的尺寸，默认为5
        sigma:高斯分布的方差，默认为1
    
    Returns
    -------
        kernel:返回的高斯核
    """
    kernel = np.zeros(shape=(kernel_size), dtype=np.float)
    radius = kernel_size // 2
    for x in range(-radius, radius+1):
        v = 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x/sigma) ** 2))
        kernel[x+radius] = v
    
    return kernel

def Gaussian_smoothing_1D(source_signal, ksize=5, sigma=1):
    """
    1维高斯平滑

    先调用create_gaussian_kernel_1D函数生成高斯核，然后再使用卷积的方案，进行平滑

    Parameters
    ----------
        source_signal:原始信号
        ksize:高斯核的尺寸
        sigma:高斯分布的方差
    
    Returns
    -------
        outputs:平滑后的数据
    """
    kernel = create_gaussian_kernal_1D(kernel_size=ksize, sigma=sigma)
    w = len(source_signal)
    w_pad = (ksize - 1) // 2

    # 原始信号padding
    source_signal = np.pad(source_signal, pad_width=(w_pad, w_pad), mode='constant', constant_values=0)
    outputs = np.zeros(shape=(w))
    for i in range(w):
        outputs[i] = np.sum(np.multiply(source_signal[i:i+ksize], kernel))
    
    return outputs

def cal_power(signal):
    """
    计算 1D 信号的能量值
    对signal做平方累计后，求均值并开方。

    Parameters
    ----------
        signal:目标计算信号

    Returns
    -------
        power:该段信号的能量值

    """
    power = 0
    for i in range(len(signal)):
        power += signal[i] ** 2
    
    power = power / len(signal)
    power = math.sqrt(power)
    return power

# def delete_noise_peak(signal):
#     size = len(signal)
#     index = 0
    
#     while index < size:
#         if signal[index] > 0:
#             begin = index
#             index += 1
#             while True:
#                 if index >= size:
#                     break
#                 if signal[index] == 0:
#                     break
#                 index += 1
#             print(index-begin)
#             if (index - begin) < 9:
#                 signal[begin:index] = 0
#         index += 1
#     return signal

def delete_noise_peak(signal):
    size = len(signal)
    peaks = []
    index = 0
    while index < size:
        if signal[index] > 0:
            # get the begin of peak
            peak_value = signal[index]
            begin = index
            # get the end of peak
            while True:
                index += 1
                if index == size or signal[index] == 0:
                    break
                if signal[index] > peak_value:
                    peak_value = signal[index]
            if index == size:
                end = size - 1
            else:
                end = index
            peaks.append([begin, end, peak_value])
        else:      
            index += 1

    # if only one peak, return
    # if len(peaks) == 1:
    #     return signal

    for peak in peaks:
        begin = peak[0]
        end = peak[1]
        value = peak[2]

        if value <= 3:
            signal[begin:end] = 0
            continue
        
        while signal[begin] <= 1:
            begin += 1
        
        while signal[end] <= 1:
            end -= 1

        if value <= 13 and (end - begin) <= 9:
            signal[peak[0]:peak[1]] = 0


    return signal

def delete_peak(signal):
    size = len(signal)
    peaks = []
    index = 0
    while index < size:
        if signal[index] > 0:
            # get the begin of peak
            peak_value = signal[index]
            begin = index
            # get the end of peak
            while True:
                index += 1
                if index == size or signal[index] == 0:
                    break
                if signal[index] > peak_value:
                    peak_value = signal[index]
            if index == size:
                end = size - 1
            else:
                end = index
            peaks.append([begin, end, peak_value])
        else:      
            index += 1
    
    peak_index = []

    for peak in peaks:
        begin = peak[0]
        end = peak[1]
        value = peak[2]

        if value <= 3:
            # signal[begin:end] = 0
            continue

        # if (peak[0] >= 50 and peak[0] <= 60) or (peak[1] >= 50 and peak[1] <= 60):
        #     continue 

        while signal[begin] <= 1:
            begin += 1
        
        while signal[end] <= 1:
            end -= 1

        if value <= 11 and (end - begin) <= 5:
            peak_index.append([peak[0],peak[1]])

    return peak_index


def cal_main_fq(signal, t):
    """
    根据频域向量的最大值，计算该段信号的主频率

    1. 先找到该段频域数据中大于 0 的数据的index_greater_zero，
        若没有大于0的数据，则说明该段数据没有动作影响，主频率为19000

        若有大于0的数据，则说明该段数据有动作影响。
            (1). 计算该段频域数据幅度最大的频率分量
            (2). 如果该频率分量与19000的关系（大于或小于）
                若小于，则返回index_greater_zero[0]对应的频率
                若大于，则返回index_greater_zero[-1]对应的频率
    
    Parameters
    ----------
        signal:幅度值向量
        t:频率值向量
    Returns
    -------
        main_fq:主频率
    """
    main_fq = 19000

    index_greater_zero = np.where(signal>0)[0]

    if(len(index_greater_zero) == 0):
        return main_fq

    max_index = np.argmax(signal)
    max_fq = t[max_index]

    if max_fq < 19000:
        # main_fq = t[index_greater_zero[0] - 1]
        if index_greater_zero[0] - 1 < 0:
            main_fq = t[0]
        else:
            main_fq = t[index_greater_zero[0] - 1]
    else:
        if index_greater_zero[-1] + 1 >= len(t):
            main_fq = t[-1]
        else:
            main_fq = t[index_greater_zero[-1] + 1]

    return main_fq

def motion_detection(signal, FS=44100 ,npreseg=8192, noverlap=7680, window_size=10, stride=5, threshold_1=3, threshold_2=14):
    """
    动作检测函数

    该函数能够实现对一段时序数据，检测其中有动作的时间段，并将时间段的起始和终止时间点，
    以数据对的形式返回

    Parameters
    ----------
        signal:带有动作的时序信号
        FS:数据的采样率                                        默认为44100
        npreseg:对时序信号进行STFT时的窗口大小                  默认为8192
        noverlap:对时序信号进行STFT时，窗口移动的重叠大小        默认为7680
        window_size:进行动作检测时的窗口大小                    默认为10
        stride:在进行动作检测时，窗口移动的步长                  默认为5

    Returns
    -------
        motion_time_index:动作时间段的始末点对集合
    
    >>> motion_time_index = motion_detection(signal, 44100, 8192, 7680, 10, 5)

    """
    # 调用 audio2spetrum 函数进行STFT
    f, t, PP = audio2spetrum(signal, npreseg, noverlap, FS, True)
    
    # 截取【18700 19300】频段之间的数据
    lower_boundary = np.where(f>=18695)[0][0]
    upper_boundary = np.where(f<=19305)[0][-1]
    PP = PP[lower_boundary:upper_boundary+1, :]
    f = f[lower_boundary: upper_boundary+1]

    # 调用cal_main_fq函数，计算每个窗口的主频率, 形成 1 维主频率向量
    shape = PP.shape
    main_fq_each_window = np.zeros(shape=shape[1])
    for j in range(shape[1]):
        current_window = PP[:,j]
        current_window = Gaussian_smoothing_1D(current_window, ksize=5, sigma=1)
        current_window[current_window<0] = 0
        # 手机数据时使用
        # current_window = delete_noise_peak(current_window)
        main_fq_each_window[j] = cal_main_fq(current_window, f)


    # 使用窗口累积的方法，截取出有信号变化的时间段，result_index中每一行为一个信号变化时间段
    # 第n列开始有动作，则表示在(n*(npreseg-noverlap))的位置开始有信号。第m列开始没有动作，则表示在（m*(npreseg-noverlap）的位置截至
    index = 0
    result_index = [[0,0]]
    
    # 阈值1，用于比较两个时间片之间的间隔，若两个时间片间的间隔小于该阈值，则说明
    # 两个时间片属于同一个动作影响内，应该合并。
    # 该阈值设置为0.5秒
    # threshold_1 = 13      # word
    # threshold_1 = 3        # letter

    while((index+window_size) < shape[1]):
        power = cal_power(main_fq_each_window[index:index+window_size])
        if(abs(power - 19000) > 0):
            # 若窗口的能量-19000的绝对值大于0，则说明该窗口符合条件，作为起始点 begin_index
            begin_index = index
            end_index = index
            while(True):
                if(index+window_size > shape[1]):
                    end_index = index
                    break

                temp_power = cal_power(main_fq_each_window[index : index+window_size])
                if(abs(temp_power - 19000) == 0):
                    # 当窗口能量-19000小于或等于0，则说明窗口不符合条件，作为结束点 end_index
                    end_index = index
                    break
                index += stride
            

            # while(begin_index + 1 < end_index and main_fq_each_window[begin_index+1] == 19000):
            #     begin_index+=1
            # while(end_index -1 > begin_index and main_fq_each_window[end_index-1] == 19000):
            #     end_index-=1
            # 比较当前这个时间段的起始点和上一个时间段的终止点
            # 若两点之间间隔太小，则表示这两个时间段是处于同一个动作内的
            compare_index1 = result_index[-1][1]
            compare_index2 = begin_index
            if (compare_index2 - compare_index1) < threshold_1:
                result_index[-1][1] = end_index
            else:
                result_index.append([begin_index, end_index])    
        index += stride

    # 去除持续时间太短的时间间隔
    # 当前时间段持续时间太小，则说明该动作不是一个书写单词的动作
    # 设置阈值2为1.2秒
    motion_time_index = []
    # threshold_2 = 80 # word
    # threshold_2 = 14   # letter
    for i in range(len(result_index)):
        if(result_index[i][1] - result_index[i][0] > threshold_2):
            # begin_time_index = result_index[i][0] * (npreseg-noverlap)
            # end_time_index = result_index[i][1] * (npreseg-noverlap)
            # motion_time_index.append([begin_time_index, end_time_index])
            motion_time_index.append([result_index[i][0], result_index[i][1]])

    return main_fq_each_window, motion_time_index

def split_word(root_path):
    paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            pt = root + "/" + file
            if pt.endswith(".pcm") or pt.endswith(".wav"):
                paths.append(pt)

    paths.sort()
    datas = []
    for pt in paths:
        if pt.endswith(".pcm"):
            datas.append(DataIO.read_pcm(pt))
        elif pt.endswith(".wav"):
            datas.append(DataIO.read_wav(pt))

    words = []
    rets = []
    for i in range(len(datas)):
        signal = datas[i]
        signal = filter(signal, 4, [18985, 19015], 44100, 'bandstop')
        main_fq, motion_time_index = motion_detection(signal=signal, FS=44100,npreseg=8192,noverlap=7680, window_size=5, stride=2, threshold_1=40, threshold_2=70)

        f, t, PP = audio2spetrum(signal, 8192, 7680, 44100, True)
        # 截取【18700 19300】频段之间的数据
        lower_boundary = np.where(f>=18695)[0][0]
        upper_boundary = np.where(f<=19305)[0][-1]
        PP = PP[lower_boundary:upper_boundary+1, :]
        f = f[lower_boundary: upper_boundary+1]

        word = []
        ts = []
        for j in range(len(motion_time_index)):
            sub_PP = PP[:, motion_time_index[j][0]:motion_time_index[j][1]]
            sub_PP = matrix_scaler(sub_PP)
            sub_PP = Gaussian_smoothing_2D(sub_PP, ksize=5, sigma=1)
            sub_PP = filter_noise(sub_PP, 0.68)

            word.append(sub_PP)
            # sub_t = t[motion_time_index[j][0]:motion_time_index[j][1]]
            # ts.append(sub_t)


        words.append(word)      # for diff pattern
        # rets.append(ts)

    return words, f


# def split_word(root_path, save_path):
#     paths = []
#     for root, dirs, files in os.walk(root_path):
#         for file in files:
#             pt = root + "/" + file
#             if pt.endswith(".pcm"):
#                 paths.append(pt)
    
#     # users = []
#     labels = []
#     datas = []
#     for pt in paths:
#         ptl = pt.split("/")
#         # users.append(ptl[-3])
#         labels.append(ptl[-2])
#         datas.append(DataIO.read_pcm(pt))
    

#     # """有效信号截取"""
#     """方案2 用窗口内的能量最大值index作为基准进行动作检测"""
#     words = []
#     word_labels = []
#     # word_users = []
#     for i in range(len(datas)):
#         signal = datas[i]
#         signal = filter(signal, 4, [18985, 19015], 44100, 'bandstop')

#         _, motion_time_index = motion_detection(signal=signal, FS=44100,npreseg=8192,noverlap=7680, window_size=5, stride=2, threshold_1=50, threshold_2=150)
#         # print(len(motion_time_index))
#         # 将原始数据进行切分成单个单词
#         for j in range(len(motion_time_index)):
#             start_index = int(motion_time_index[j][0] - 44100 // 6)
#             end_index = int(motion_time_index[j][1] + 44100 // 5)
#             if start_index < 0:
#                 start_index = 0
#             if end_index > len(signal):
#                 end_index = len(signal)-1
        
#             words.append(signal[start_index : end_index])
#             word_labels.append(labels[i])
#             # word_users.append(users[i])

#     count = 0
#     lb = word_labels[0]
#     for i in range(len(words)):
#         word = words[i]
#         label = word_labels[i]
#         # user = word_users[i]
#         if lb != label:
#             count = 1
#             lb = label
#         else:
#             count += 1

#         if os.path.exists(save_path) is False:
#             os.makedirs(save_path)

#         np.save(save_path + "/" + label + "_" + str(count) + ".npy", word)        
#         # if os.path.exists(save_path + "/" + user) is False:
#         #     os.makedirs(save_path + "/" + user)

#         # np.save(save_path + "/" + user + "/" + label + "_" + str(count) + ".npy", word)

#     print("split word successful!")


def check_split_word(root_path):
    paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            pt = root + "/" + file
            if pt.endswith(".pcm") or pt.endswith(".wav"):
                paths.append(pt)


    paths.sort()
    datas = []
    for pt in paths:
        if pt.endswith(".pcm"):
            datas.append(DataIO.read_pcm(pt))
        elif pt.endswith(".wav"):
            datas.append(DataIO.read_wav(pt))
    

    # """有效信号截取"""
    """方案2 用窗口内的能量最大值index作为基准进行动作检测"""
    for i in range(len(datas)):
        signal = datas[i]
        signal = filter(signal, 4, [18985, 19015], 44100, 'bandstop')
        main_fq, motion_time_index = motion_detection(signal=signal,FS=44100, npreseg=8192, noverlap=7680, window_size=5, stride=2, threshold_1=50, threshold_2=70)

        f, t, PP = audio2spetrum(signal, 8192, 7680, 44100, True)

        # 截取【18700 19300】频段之间的数据
        lower_boundary = np.where(f>=18695)[0][0]
        upper_boundary = np.where(f<=19305)[0][-1]
        PP = PP[lower_boundary:upper_boundary+1, :]
        f = f[lower_boundary: upper_boundary+1]

        # for j in range(210, 224):
        #     print("col " + str(j))
        #     current_window = PP[:,j]
        #     current_window = Gaussian_smoothing_1D(current_window, ksize=5, sigma=1)
        #     current_window[current_window<0] = 0
        #     current_window = delete_noise_peak(current_window)
        #     # plt.figure()
        #     # plt.plot(current_window)
        #     # plt.savefig("../img/ZPZ/" + str(j) + ".png")
        #     # plt.close()

        # exit(0)


        plt.figure(figsize=(10, 4))
        plt.subplot(2,1,1)
        plt.title(paths[i])
        plt.pcolormesh(t, f, PP)
        plt.ylim([18700, 19300])
        plt.axis("off")
        # plt.colorbar()
        plt.subplot(2,1,2)
        plt.plot(main_fq)
        plt.xlim([0, len(main_fq)])
        for j in range(len(motion_time_index)):
            plt.axvline(motion_time_index[j][0], linestyle="--", color="black")
            plt.axvline(motion_time_index[j][1], linestyle="--", color="green")
        
        # plt.show()
        plt.savefig("../img/check/" + paths[i].split("/")[-1][:-4] + ".png")
        plt.close()

def split_letter(root_path):
    paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            pt = root + "/" + file
            if pt.endswith(".pcm") or pt.endswith(".wav"):
                paths.append(pt)

    paths.sort()
    # uid = []
    datas = []
    for pt in paths:
        # ptl = pt.split("/")
        # uid.append(ptl[-4])
        if pt.endswith(".pcm"):
            datas.append(DataIO.read_pcm(pt))
        elif pt.endswith(".wav"):
            datas.append(DataIO.read_wav(pt))


    letters = []
    resets = []
    for i in range(len(datas)):
        signal = datas[i]
        signal = filter(signal, 4, [18985, 19015], 44100, 'bandstop')
        main_fq, motion_time_index = motion_detection(signal=signal, FS=44100,npreseg=8192,noverlap=7680, window_size=5, stride=2, threshold_1=25, threshold_2=5)

        f, t, PP = audio2spetrum(signal, 8192, 7680, 44100, True)
        # 截取【18700 19300】频段之间的数据
        lower_boundary = np.where(f>=18695)[0][0]
        upper_boundary = np.where(f<=19305)[0][-1]
        PP = PP[lower_boundary:upper_boundary+1, :]
        f = f[lower_boundary: upper_boundary+1]

        letter = []
        reset = []
        for j in range(len(motion_time_index)):
            # sub_PP = PP[:, motion_time_index[j][0]:motion_time_index[j][1]]
            # sub_PP = matrix_scaler(sub_PP)
            # sub_PP = Gaussian_smoothing_2D(sub_PP, ksize=5, sigma=1)
            # sub_PP = filter_noise(sub_PP, 0.68)
            # letter.append(sub_PP)
            if (j % 2 == 1):
                # letter.append(PP_filter_noise[:,motion_time_index[j][0]:motion_time_index[j][1]])
                sub_PP = PP[:, motion_time_index[j][0]:motion_time_index[j][1]]
                # sub_PP = matrix_scaler(sub_PP)
                # sub_PP = Gaussian_smoothing_2D(sub_PP, ksize=5, sigma=1)
                # sub_PP = filter_noise(sub_PP, 0.68)
                letter.append(sub_PP)
            else:
                sub_PP = PP[:, motion_time_index[j][0]:motion_time_index[j][1]]
                # sub_PP = matrix_scaler(sub_PP)
                # sub_PP = Gaussian_smoothing_2D(sub_PP, ksize=5, sigma=1)
                # sub_PP = filter_noise(sub_PP, 0.68)
                reset.append(sub_PP)

        letters.append(letter)      # for diff pattern
        resets.append(reset)
    # save_letter = np.array(letters[0])
    # count = letters[0].shape[1]
    # indexs = [np.array([0, count])]

    # for i in range(1, len(letters)):
    #     letter = letters[i]
    #     indexs.append(np.array([count, count + letter.shape[1]]))
    #     count+=letter.shape[1]
    #     save_letter = np.concatenate((save_letter, letter), axis=1)
    
    # indexs = np.array(indexs)

    # return save_letter, indexs
    # return letters, resets
    return letters, resets, f

def check_split_letter(root_path):
    paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            pt = root + "/" + file
            if pt.endswith(".pcm") or pt.endswith(".wav"):
                paths.append(pt)

    paths.sort()
    datas = []
    for pt in paths:
        if(pt.endswith(".pcm")):
            datas.append(DataIO.read_pcm(pt))
        elif(pt.endswith(".wav")):
            datas.append(DataIO.read_wav(pt))

    for i in range(len(datas)):
        signal = datas[i]
        signal = filter(signal, 4, [18985, 19015], 44100, 'bandstop')
        main_fq, motion_time_index = motion_detection(signal=signal,FS=44100, npreseg=8192, noverlap=7680, window_size=5, stride=2, threshold_1=25, threshold_2=5)


        f, t, PP = audio2spetrum(signal, 8192, 7680, 44100, True)
        # 截取【18700 19300】频段之间的数据
        lower_boundary = np.where(f>=18695)[0][0]
        upper_boundary = np.where(f<=19305)[0][-1]
        PP = PP[lower_boundary:upper_boundary+1, :]
        f = f[lower_boundary: upper_boundary+1]

        # 频谱图归一化
        PP_norm = matrix_scaler(PP)

        # 频谱图平滑
        PP_gauss_smo = Gaussian_smoothing_2D(PP_norm, ksize=5, sigma=1)

        # 去除噪声
        # PP_filter_noise = filter_noise(PP_gauss_smo, 0.72)
        PP_filter_noise = filter_noise(PP_gauss_smo, 0.68)

        # PP_filter_noise_smo = Gaussian_smoothing_2D(PP_filter_noise, ksize=3, sigma=1)
        plt.figure(figsize=(18, 8))
        plt.subplot(2,1,1)
        plt.title(paths[i])
        plt.pcolormesh(t, f, PP_filter_noise)
        plt.ylim([18700, 19300])
        plt.axis("off")
        # plt.colorbar()
        plt.subplot(2,1,2)
        plt.plot(main_fq)
        plt.xlim([0, len(main_fq)])
        for j in range(len(motion_time_index)):
            # if(j % 2 == 1):
            # print(str(j) +  " : ")
            # print(motion_time_index[j][1] - motion_time_index[j][0])
            plt.axvline(motion_time_index[j][0], linestyle="--", color="black")
            plt.axvline(motion_time_index[j][1], linestyle="--", color="green")
        
        # plt.show()
        plt.savefig("../img/check/" + paths[i].split("/")[-1][:-4] + ".png")

def split_reset_action(root_path):
    paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            pt = root + "/" + file
            if pt.endswith(".pcm") or pt.endswith(".wav"):
                paths.append(pt)
    paths.sort()
    datas = []
    for pt in paths:
        if pt.endswith(".pcm"):
            datas.append(DataIO.read_pcm(pt))
        elif pt.endswith(".wav"):
            datas.append(DataIO.read_wav(pt))

    letters = []
    for i in range(len(datas)):
        signal = datas[i]
        signal = filter(signal, 4, [18985, 19015], 44100, 'bandstop')
        main_fq, motion_time_index = motion_detection(signal, 44100, 8192, 7680, 5, 2)

        f, t, PP = audio2spetrum(signal, 8192, 7680, 44100, True)
        # 截取【18700 19300】频段之间的数据
        lower_boundary = np.where(f>=18695)[0][0]
        upper_boundary = np.where(f<=19305)[0][-1]
        PP = PP[lower_boundary:upper_boundary+1, :]
        f = f[lower_boundary: upper_boundary+1]

        # 频谱图归一化
        # PP_norm = matrix_scaler(PP)

        # # 频谱图平滑
        # PP_gauss_smo = Gaussian_smoothing_2D(PP_norm, ksize=5, sigma=1)

        # # 去除噪声
        # PP_filter_noise = filter_noise(PP_gauss_smo, 0.72)

        letter = []
        # motion_time_index = motion_time_index[1:]
        for j in range(3, len(motion_time_index), 4):
            sub_PP = PP[:, motion_time_index[j][0]:motion_time_index[j][1]]
            sub_PP = matrix_scaler(sub_PP)

            # 频谱图平滑
            sub_PP = Gaussian_smoothing_2D(sub_PP, ksize=5, sigma=1)

            # 去除噪声
            sub_PP = filter_noise(sub_PP, 0.68)
            letter.append(sub_PP)
        
        # letters.append(letter)
        letters += letter
    
    save_letter = np.array(letters[0])
    count = letters[0].shape[1]
    indexs = [np.array([0, count])]
    for i in range(1, len(letters)):
        letter = letters[i]
        indexs.append(np.array([count, count + letter.shape[1]]))
        count+=letter.shape[1]
        save_letter = np.concatenate((save_letter, letter), axis=1)
    
    indexs = np.array(indexs)
    # print(save_letter.shape)
    # print(indexs)
    # exit(0)
    return save_letter, indexs

def check_preprocess(signal, nperseg, noverlap, FS, log_flag, pic_path):
    # STFT，转化为频谱图
    f, t, PP = audio2spetrum(signal, nperseg, noverlap, FS, log_flag)       # 步长 512

    # 去除低频段的背景噪声分量
    lower_boundary = np.where(f>=18695)[0][0]
    upper_boundary = np.where(f<=19305)[0][-1]
    PP = PP[lower_boundary:upper_boundary+1, :]
    f = f[lower_boundary: upper_boundary+1]

    # 频谱图归一化
    PP_norm = matrix_scaler(PP)

    # 频谱图平滑
    PP_gauss_smo = Gaussian_smoothing_2D(PP_norm, ksize=5, sigma=1)

    # 去除噪声
    PP_filter_noise = filter_noise(PP_gauss_smo, 0.70)

    plt.figure(figsize=(12, 8))
    plt.subplot(2,2,1)
    plt.set_cmap('parula')
    plt.pcolormesh(t, f, PP)
    plt.ylim([18700, 19300])
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.set_cmap('parula')
    plt.pcolormesh(t, f, PP_gauss_smo)
    plt.ylim([18700, 19300])
    plt.axis("off")

    plt.subplot(2,2,3)
    plt.set_cmap("parula")
    plt.pcolormesh(t, f, PP_filter_noise)
    plt.ylim([18700, 19300])
    plt.axis("off")

    # plt.show()
    plt.savefig(pic_path)

def preprocess(signal, nperseg, noverlap, FS, log_flag, flip_flag=True):
    # STFT，转化为频谱图
    f, t, PP = audio2spetrum(signal, nperseg, noverlap, FS, log_flag)       # 步长 512

    # 去除低频段的背景噪声分量
    lower_boundary = np.where(f>=18695)[0][0]
    upper_boundary = np.where(f<=19305)[0][-1]
    PP = PP[lower_boundary:upper_boundary+1, :]
    f = f[lower_boundary: upper_boundary+1]

    # 频谱图归一化
    PP_norm = matrix_scaler(PP)

    # 频谱图平滑
    PP_gauss_smo = Gaussian_smoothing_2D(PP_norm, ksize=5, sigma=1)

    # 去除噪声
    PP_filter_noise = filter_noise(PP_gauss_smo, 0.68)

    if flip_flag == True:
        PP_filter_noise = np.flip(PP_filter_noise, axis=0)

    return f, t, PP_filter_noise

def forpaper():
    root_path = "/sdo/zcl/AcouWrite/DATA/word/ZCL/WORD/AM/2022-09-22_15h-10m-44s.wav"
    signal = DataIO.read_wav(root_path)
    signal_after = filter(signal, 4, [18985, 19015], 44100, 'bandstop')
    main_fq, motion_time_index = motion_detection(signal=signal_after, FS=44100,npreseg=8192,noverlap=7680, window_size=5, stride=2, threshold_1=50, threshold_2=70)

    index = 1
    for begin,end in motion_time_index:
        sub_signal = signal[begin:end+22050]
        sub_signal = sub_signal/32767
        np.save(os.path.join("/sdo/zcl/AcouWrite/DATA/forpaper/AM",str(index+10)+".npy"), sub_signal)
        index += 1
    # print(motion_time_index)

# forpaper()

def main(): 
    path = "/sdo/zcl/AcouWrite/DATA/forpaper/shuoming/pinban"
    check_split_word(path)
#     root_path = "/sdo/zcl/AcouWrite/DATA/word/LCZ/WORD_Xiaomi/PERSON/2023-02-21_17h-14m-23s.wav"
#     signal = DataIO.read_wav(root_path)
#     signal = filter(signal, 4, [18985, 19015], 44100, 'bandstop')

#     f, t, PP = audio2spetrum(signal, 8192, 7680, 44100, True)
#     # 截取【18700 19300】频段之间的数据
#     lower_boundary = np.where(f>=18695)[0][0]
#     upper_boundary = np.where(f<=19305)[0][-1]
#     PP = PP[lower_boundary:upper_boundary+1, :]
#     f = f[lower_boundary: upper_boundary+1]

#     for i in range(105,130):
#         current_window = PP[:,i]
#         current_window = Gaussian_smoothing_1D(current_window, ksize=5, sigma=1)
#         current_window[current_window<0] = 0
#         current_window = delete_noise_peak(current_window)
#         plt.figure()
#         plt.plot(current_window)
#         plt.savefig("../img/check/"+str(i)+".png")
#         plt.close()
#     # check_split_letter(root_path)
#     # check_split_word(root_path)
#     # exit(0)



# main()