import numpy as np
import os
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from Process import SignalProcess, DataIO, Display
import cv2
import torchaudio.transforms as T
import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import colors as c

import numpy as np
import scipy.signal as sig

from matplotlib import rcParams
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False


# data = np.load()


# data1 = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/shuoming/data/pinban_2.npy")
# data2 = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/shuoming/data/sanxing_3.npy")
# data3 = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/shuoming/data/xiaomi_2.npy")

# f = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/figure3-5f.npy")
# t = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/figure3-5t.npy")

# plt.figure(figsize=(12, 8))
# plt.set_cmap('parula')
# plt.pcolormesh(t[:data1.shape[1]], f, data1)
# plt.ylim([18700, 19300])
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.savefig("./img/paper/figure4-16a.jpg", dpi=600)
# plt.close()

# plt.figure(figsize=(12, 8))
# plt.set_cmap('parula')
# plt.pcolormesh(t[:data2.shape[1]], f, data2)
# plt.ylim([18700, 19300])
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.savefig("./img/paper/figure4-16b.jpg", dpi=600)
# plt.close()

# plt.figure(figsize=(12, 8))
# plt.set_cmap('parula')
# plt.pcolormesh(t[:data3.shape[1]], f, data3)
# plt.ylim([18700, 19300])
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.savefig("./img/paper/figure4-16c.jpg", dpi=600)
# plt.close()
# exit()


# # 图3-3
# data = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/figure3-5.npy")
# # data = DataIO.read_pcm("/sdo/zcl/AcouWrite/DATA/forpaper/shuoming/pinban/2023-02-02_21h-45m-30s.pcm")
# # data = SignalProcess.filter(data, 4, [18985, 19015], 44100, 'bandstop')
# # f, t, data = SignalProcess.audio2spetrum(data, 8192, 7680, 44100, True)
# f = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/figure3-5f.npy")
# t = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/figure3-5t.npy")


# data = SignalProcess.matrix_scaler(data)
# data = SignalProcess.Gaussian_smoothing_2D(data, ksize=5, sigma=1)
# data = SignalProcess.filter_noise(data, 0.68)

# fig = plt.figure(figsize=(12, 8))
# axe = fig.add_subplot(111)
# # axe.grid(True)
# plt.set_cmap('parula')
# plt.pcolormesh(t, f, data)
# # axe.plot(window,linewidth=3.0) 
# axe.set_ylabel("频率 (Hz)", fontsize=33, labelpad=15) 
# axe.set_xlabel("时间 (秒)", fontsize=33, labelpad=15) 
# axe.tick_params(labelsize=25)
# fig.savefig("./img/paper/ppt_detect.jpg", dpi=600, bbox_inches="tight")
# plt.close()

# plt.figure()
# plt.set_cmap('parula')
# plt.pcolormesh(t[40:], f, data[:,40:])
# plt.ylim([18700, 19300])
# # plt.tick_params(labelsize=20)
# # plt.xlabel("Time (s)", fontsize=20)  # here
# # plt.ylabel("Frequency (Hz)", fontsize=20)
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.savefig("./img/paper/ppt_detect.jpg", dpi=600)
# plt.close()
# exit()

# # 不同速度和书写大小的A 
# diff_size_1 = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_size_1.npy")
# diff_size_1_t = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_size_1t.npy")
# diff_size_1 = SignalProcess.matrix_scaler(diff_size_1)
# diff_size_1 = SignalProcess.Gaussian_smoothing_2D(diff_size_1, ksize=5, sigma=1)
# diff_size_1 = SignalProcess.filter_noise(diff_size_1, 0.68)


# diff_size_2 = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_size_2.npy")
# diff_size_2_t = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_size_2t.npy")
# diff_size_2 = SignalProcess.matrix_scaler(diff_size_2)
# diff_size_2 = SignalProcess.Gaussian_smoothing_2D(diff_size_2, ksize=5, sigma=1)
# diff_size_2 = SignalProcess.filter_noise(diff_size_2, 0.68)


# diff_size_3 = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_size_3.npy")
# diff_size_3_t = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_size_3t.npy")
# diff_size_3 = SignalProcess.matrix_scaler(diff_size_3)
# diff_size_3 = SignalProcess.Gaussian_smoothing_2D(diff_size_3, ksize=5, sigma=1)
# diff_size_3 = SignalProcess.filter_noise(diff_size_3, 0.68)





# diff_speed_1 = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_speed_1.npy")
# diff_speed_1 = diff_speed_1[:, :-20]
# diff_speed_1 = np.pad(diff_speed_1, ((0,0),(20, 0)), mode="constant")
# diff_speed_1_t = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_speed_1t.npy")
# diff_speed_1 = SignalProcess.matrix_scaler(diff_speed_1)
# diff_speed_1 = SignalProcess.Gaussian_smoothing_2D(diff_speed_1, ksize=5, sigma=1)
# diff_speed_1 = SignalProcess.filter_noise(diff_speed_1, 0.68)

# diff_speed_2 = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_speed_2.npy")
# diff_speed_2_t = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_speed_2t.npy")
# diff_speed_2 = SignalProcess.matrix_scaler(diff_speed_2)
# diff_speed_2 = SignalProcess.Gaussian_smoothing_2D(diff_speed_2, ksize=5, sigma=1)
# diff_speed_2 = SignalProcess.filter_noise(diff_speed_2, 0.68)

# diff_speed_3 = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_speed_3.npy")
# diff_speed_3_t = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_speed_3t.npy")
# diff_speed_3 = SignalProcess.matrix_scaler(diff_speed_3)
# diff_speed_3 = SignalProcess.Gaussian_smoothing_2D(diff_speed_3, ksize=5, sigma=1)
# diff_speed_3 = SignalProcess.filter_noise(diff_speed_3, 0.68)



# f = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/figure3-5f.npy")


# plt.figure(figsize=(8, 8))
# plt.set_cmap("parula")
# plt.pcolormesh(diff_size_1_t, f, diff_size_1)
# plt.tick_params(labelsize=20)
# plt.xlabel("Time (s)", fontsize=20)  # here
# plt.ylabel("Frequency (Hz)", fontsize=20)
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# # plt.savefig("./img/paper/diff/diff_size_1.jpg", dpi=600,bbox_inches='tight')
# plt.savefig("./img/paper/diff/diff_size_1_s.jpg", dpi=600)
# plt.close()


# plt.figure(figsize=(8, 8))
# plt.set_cmap("parula")
# plt.pcolormesh(diff_size_1_t, f, diff_size_2)
# plt.tick_params(labelsize=20)
# plt.xlabel("Time (s)", fontsize=20)  # here
# plt.ylabel("Frequency (Hz)", fontsize=20)
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)

# # plt.savefig("./img/paper/diff/diff_size_2.jpg", dpi=600,bbox_inches='tight')
# plt.savefig("./img/paper/diff/diff_size_2_s.jpg", dpi=600)
# plt.close()

# plt.figure(figsize=(8, 8))
# plt.set_cmap("parula")
# plt.pcolormesh(diff_size_1_t, f, diff_size_3)
# plt.tick_params(labelsize=20)
# plt.xlabel("Time (s)", fontsize=20)  # here
# plt.ylabel("Frequency (Hz)", fontsize=20)
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.savefig("./img/paper/diff/diff_size_3_s.jpg", dpi=600)
# # plt.savefig("./img/paper/diff/diff_size_3.jpg", dpi=600,bbox_inches='tight')
# plt.close()

# plt.figure(figsize=(8, 8))
# plt.set_cmap("parula")
# plt.pcolormesh(diff_speed_1_t, f, diff_speed_1)
# plt.tick_params(labelsize=20)
# plt.xlabel("Time (s)", fontsize=20)  # here
# plt.ylabel("Frequency (Hz)", fontsize=20)
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)

# # plt.savefig("./img/paper/diff/diff_speed_1.jpg", dpi=600,bbox_inches='tight')
# plt.savefig("./img/paper/diff/diff_speed_1_s.jpg", dpi=600)
# plt.close()

# plt.figure(figsize=(8, 8))
# plt.set_cmap("parula")
# plt.pcolormesh(diff_speed_1_t, f, diff_speed_2)
# plt.tick_params(labelsize=20)
# plt.xlabel("Time (s)", fontsize=20)  # here
# plt.ylabel("Frequency (Hz)", fontsize=20)
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)

# plt.savefig("./img/paper/diff/diff_speed_2_s.jpg", dpi=600)
# # plt.savefig("./img/paper/diff/diff_speed_2.jpg", dpi=600,bbox_inches='tight')
# plt.close()

# plt.figure(figsize=(8, 8))
# plt.set_cmap("parula")
# plt.pcolormesh(diff_speed_1_t, f, diff_speed_3)
# plt.tick_params(labelsize=20)
# plt.xlabel("Time (s)", fontsize=20)  # here
# plt.ylabel("Frequency (Hz)", fontsize=20)
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)

# plt.savefig("./img/paper/diff/diff_speed_3_s.jpg", dpi=600)
# # plt.savefig("./img/paper/diff/diff_speed_3.jpg", dpi=600,bbox_inches='tight')
# plt.close()

# exit(0)


# 信号预处理
# path = "/sdo/zcl/AcouWrite/DATA/forpaper/AM/11.npy"

# def preprocessing(wav_data):    
#     BPsignal = wav_data

#     # 带阻
#     [d, c] = sig.butter(4, [18985 / 44100 * 2, 19015 / 44100 * 2], 'bandstop')
#     BPsignal = sig.filtfilt(d, c, BPsignal)

#     nfft = 8192
#     overlap = 7680
#     step = nfft - overlap
#     # [f, t, Zxx] = sig.stft(BPsignal, 44100,window='hamm', nperseg=nfft, noverlap=overlap, detrend=False)
#     [f, t, Zxx] = sig.spectrogram(BPsignal, 44100, window="hamm", nperseg=nfft, noverlap=overlap, detrend=False)

#     Zxx_magnitude = np.abs(Zxx)
#     PP = 10 * np.log10(Zxx_magnitude + 2.2204e-16)
#     return f, t, PP

# data = np.load(path)
# f, t, data = preprocessing(data)
# data = data[:,:-25]
# t= t[:-25]
# plt.figure()
# plt.set_cmap('parula')
# plt.pcolormesh(t, f, data)
# plt.ylim([18700, 19300])
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# # plt.colorbar()
# plt.savefig("./img/figure3-5b.jpg", dpi=600)
# plt.close()
# exit(0)


# # 不同速度和书写大小的A 
# diff_size_1 = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_size_1.npy")
# diff_size_1_t = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_size_1t.npy")

# diff_size_2 = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_size_2.npy")
# diff_size_2_t = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_size_2t.npy")

# diff_size_3 = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_size_3.npy")
# diff_size_3_t = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_size_3t.npy")

# diff_speed_1 = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_speed_1.npy")
# diff_speed_1_t = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_speed_1t.npy")

# diff_speed_2 = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_speed_2.npy")
# diff_speed_2_t = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_speed_2t.npy")

# diff_speed_3 = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_speed_3.npy")
# diff_speed_3_t = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/3-4/figure/diff_speed_3t.npy")


# f = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/figure3-5f.npy")


# plt.figure(figsize=(8, 8))
# plt.pcolormesh(diff_size_1_t, f, diff_size_1)
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)

# plt.savefig("./img/paper/diff/diff_size_1.png")
# plt.close()

# plt.figure(figsize=(8, 8))
# plt.pcolormesh(diff_size_2_t, f, diff_size_2)
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)

# plt.savefig("./img/paper/diff/diff_size_2.png")
# plt.close()

# plt.figure(figsize=(8, 8))
# plt.pcolormesh(diff_size_3_t, f, diff_size_3)
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)

# plt.savefig("./img/paper/diff/diff_size_3.png")
# plt.close()

# plt.figure(figsize=(8, 8))
# plt.pcolormesh(diff_speed_1_t, f, diff_speed_1)
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)

# plt.savefig("./img/paper/diff/diff_speed_1.png")
# plt.close()

# plt.figure(figsize=(8, 8))
# plt.pcolormesh(diff_speed_2_t, f, diff_speed_2)
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)

# plt.savefig("./img/paper/diff/diff_speed_2.png")
# plt.close()

# plt.figure(figsize=(8, 8))
# plt.pcolormesh(diff_speed_3_t, f, diff_speed_3)
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)

# plt.savefig("./img/paper/diff/diff_speed_3.png")
# plt.close()

# exit(0)

# 图3-6
# data1 = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/ZPZ/AM_2.npy")
# data2 = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/ZPZ/AM_14.npy")

# data1 = SignalProcess.matrix_scaler(data1)
# data1 = SignalProcess.Gaussian_smoothing_2D(data1, ksize=5, sigma=1)
# data1 = SignalProcess.filter_noise(data1, 0.73)
# data2 = SignalProcess.matrix_scaler(data2)
# data2 = SignalProcess.Gaussian_smoothing_2D(data2, ksize=5, sigma=1)
# data2 = SignalProcess.filter_noise(data2, 0.68)

# f = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/figure3-5f.npy")
# t = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/figure3-5t.npy")



# plt.figure(figsize=(12, 8))
# plt.set_cmap('parula')
# plt.pcolormesh(t[:data1.shape[1]], f, data1)
# plt.ylim([18700, 19300])
# # plt.tick_params(labelsize=20)
# # plt.xlabel("Time (s)", fontsize=20)  # here
# # plt.ylabel("Frequency (Hz)", fontsize=20)
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.savefig("./img/paper/figure3-6a.jpg", dpi=600)
# plt.close()
# plt.figure(figsize=(12, 8))
# plt.set_cmap('parula')
# plt.pcolormesh(t[:data2.shape[1]], f, data2)
# plt.ylim([18700, 19300])
# # plt.tick_params(labelsize=20)
# # plt.xlabel("Time (s)", fontsize=20)  # here
# # plt.ylabel("Frequency (Hz)", fontsize=20)
# plt.axis("off")
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.savefig("./img/paper/figure3-6b.jpg", dpi=600)
# plt.close()
# exit()


# 动作检测
path = "../DATA/ALL/ZCL/AM_1.npy"

data = np.load(path)
data = SignalProcess.filter(data, 4, [18985, 19015], 44100, 'bandstop')
f, t, data = SignalProcess.audio2spetrum(data, 8192, 7680, 44100, True)



y = [-40, -30, -20, -10, 0, 10, 20, 30, 40]

lower_boundary = np.where(f>=18695)[0][0]
upper_boundary = np.where(f<=19305)[0][-1]
data = data[lower_boundary:upper_boundary+1, :]
f = f[lower_boundary: upper_boundary+1]

current_window_1 = data[:,50]
current_window_2 = data[:,27]
current_window_3 = data[:,10]

current_window_1 = SignalProcess.Gaussian_smoothing_1D(current_window_1, ksize=5, sigma=1)
current_window_2 = SignalProcess.Gaussian_smoothing_1D(current_window_2, ksize=5, sigma=1)
current_window_3 = SignalProcess.Gaussian_smoothing_1D(current_window_3, ksize=5, sigma=1)

max_indx=np.argmax(current_window_1)
plt.figure()
plt.plot(f, current_window_1, linewidth=2.0)
plt.plot(f[max_indx],current_window_1[max_indx],'r-o', color="red")
plt.yticks(y)
# plt.tick_params(labelsize=15)
plt.axhline(0,linewidth=1.0, linestyle="--", color="red")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.savefig("./img/figure3-9_pos.png",dpi=600,bbox_inches='tight')
plt.close()

max_indx=np.argmax(current_window_2)
plt.figure()
plt.plot(f, current_window_2, linewidth=2.0)
plt.plot(f[max_indx],current_window_2[max_indx],'r-o', color="red")
plt.yticks(y)
plt.axhline(0,linewidth=1.0, linestyle="--", color="red")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.savefig("./img/figure3-9_neg.png",dpi=600,bbox_inches='tight')
plt.close()

max_indx=np.argmax(current_window_3)
plt.figure()
plt.plot(f, current_window_3, linewidth=2.0)
plt.plot(f[max_indx],current_window_2[max_indx],'r-o', color="red")
plt.yticks(y)
plt.axhline(0,linewidth=1.0, linestyle="--", color="red")
plt.xlabel("Frequency(Hz)")
plt.ylabel("Amplitude")
plt.savefig("./img/figure3-9_norm.png",dpi=600,bbox_inches='tight')
plt.close()
exit(0)

# 信号增强
# path = "../DATA/ALL/ZCL/AM_1.npy"

# data = np.load(path)
# data = SignalProcess.filter(data, 4, [18985, 19015], 44100, 'bandstop')
# f, t, data = SignalProcess.audio2spetrum(data, 8192, 7680, 44100, True)
# # lower_boundary = np.where(f>=18695)[0][0]
# # upper_boundary = np.where(f<=19305)[0][-1]
# # data = data[lower_boundary:upper_boundary+1, :]
# # f = f[lower_boundary: upper_boundary+1]

# t_ = t[20:]
# data_ = data[:,20:]

# data_1 = SignalProcess.matrix_scaler(data_)
# data_2 = SignalProcess.Gaussian_smoothing_2D(data_1, ksize=5, sigma=1)
# data_3 = SignalProcess.filter_noise(data_2, 0.68)


# plt.figure()
# plt.set_cmap("parula")
# plt.pcolormesh(t_, f, data_1)
# plt.ylim([18700, 19300])
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.savefig("./img/paper/figure3-8_a.jpg", dpi=600)
# plt.close()

# plt.figure()
# plt.set_cmap("parula")
# plt.pcolormesh(t_, f, data_2)
# plt.ylim([18700, 19300])
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.savefig("./img/paper/figure3-8_b.jpg", dpi=600)
# plt.close()



# plt.figure()
# plt.set_cmap("parula")
# plt.pcolormesh(t_, f, data_3)
# plt.ylim([18700, 19300])
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.savefig("./img/paper/figure3-8_c.jpg", dpi=600)
# plt.close()
# exit(0)


# 字母手势拉伸压缩

# root_path = "../DATA/letter/ZCL/after_process_v12"

# letter_signal_paths = []
# letter_index_paths = []

# for root, dir, files in os.walk(root_path):
#     for file in files:
#         if "letter" in file:
#             if "signal" in file:
#                 letter_signal_paths.append(os.path.join(root, file))
#                 # continue
#             elif "index" in file:
#                 letter_index_paths.append(os.path.join(root, file))
#                 # continue
#         else:
#             pass
# letter_signal_paths.sort()
# letter_index_paths.sort()

# diff_speed_letter_signals = []
# for i in range(len(letter_signal_paths)):

#     # letter signal 
#     letter_signal_path = letter_signal_paths[i]
#     letter_index_path = letter_index_paths[i]

#     letter_signal = np.load(letter_signal_path)
#     letter_index = np.load(letter_index_path)

#     letter = []                                          # 同一轮的数据保存为一个list，每个元素为一个ndarray，从A到Z;
#     for idx in letter_index:
#         letter.append(letter_signal[:, idx[0]:idx[1]])

#     diff_speed_letter_signals.append(letter)

# letters = diff_speed_letter_signals[0]

# letter = letters[0]
# # letter = np.flip(letter, axis=0)

# stretch = T.TimeStretch(n_freq=114)

# letter_1 = torch.from_numpy(np.ascontiguousarray(letter))
# letter_1 = np.array(torch.abs(stretch(letter_1, 1.2)))

# letter_2 = torch.from_numpy(np.ascontiguousarray(letter))
# letter_2 = np.array(torch.abs(stretch(letter_2, 0.8)))

# padsize = (letter_2.shape[1] - letter_1.shape[1])//2
# letter_1 = np.pad(letter_1, ((0,0),(padsize, padsize)), mode="constant")

# padsize = (letter_2.shape[1] - letter.shape[1])//2
# letter = np.pad(letter, ((0,0),(padsize, padsize+1)), mode="constant")

# f = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/figure3-5f.npy")
# t = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/figure3-5t.npy")

# plt.figure(figsize=(8, 8))
# plt.set_cmap("parula")
# plt.pcolormesh(t[:letter_1.shape[1]], f, letter_1)
# plt.ylim([18700, 19300])
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.savefig("./img/paper/figure3-9_a.jpg", dpi=600)
# plt.close()

# plt.figure(figsize=(8, 8))
# plt.set_cmap("parula")
# plt.pcolormesh(t[:letter.shape[1]], f, letter)
# plt.ylim([18700, 19300])
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.savefig("./img/paper/figure3-9_b.jpg", dpi=600)
# plt.close()

# plt.figure(figsize=(8, 8))
# plt.set_cmap("parula")
# plt.pcolormesh(t[:letter_2.shape[1]], f, letter_2)
# plt.ylim([18700, 19300])
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.savefig("./img/paper/figure3-9_c.jpg", dpi=600)
# plt.close()

# exit(0)

# 不同设备同一个单词AM的差异

# path1 = "/sdo/zcl/AcouWrite/DATA/forpaper/shuoming/data/sanxing_3.npy"
# path2 = "/sdo/zcl/AcouWrite/DATA/forpaper/shuoming/data/xiaomi_2.npy"
# path3 = "/sdo/zcl/AcouWrite/DATA/forpaper/shuoming/data/pinban_2.npy"

# data1 = np.load(path1)
# data2 = np.load(path2)
# data3 = np.load(path3)

# data1 = np.pad(data1, ((0,0),(14, 14)), mode="constant")
# data2 = np.pad(data2, ((0,0),(10, 10)), mode="constant")
# data3 = np.pad(data3, ((0,0),(2, 1)), mode="constant")

# plt.figure(figsize=(8,6))
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.imshow(np.flip(data1, axis=0))
# plt.savefig("./img/ZPZ/GalaxyS9.png")
# plt.close()

# plt.figure(figsize=(8,6))
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.imshow(np.flip(data2, axis=0))
# plt.savefig("./img/ZPZ/Xiaomi.png")
# plt.close()

# plt.figure(figsize=(8,6))
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.imshow(np.flip(data3, axis=0))
# plt.savefig("./img/ZPZ/Tad.png")
# plt.close()

# # print(data1.shape)
# # print(data2.shape)
# exit()



# 字母手势拉伸压缩

# root_path = "../DATA/letter/ZCL/after_process_v12"

# letter_signal_paths = []
# letter_index_paths = []
# reset_signal_paths = []
# reset_index_paths = []

# for root, dir, files in os.walk(root_path):
#     for file in files:
#         # if "LF" in file or "SF" in file:
#         #     continue
#         if "letter" in file:
#             if "signal" in file:
#                 letter_signal_paths.append(os.path.join(root, file))
#                 # continue
#             elif "index" in file:
#                 letter_index_paths.append(os.path.join(root, file))
#                 # continue
#         else:
#             if "signal" in file:
#                 reset_signal_paths.append(os.path.join(root, file))
#             elif "index" in file:
#                 reset_index_paths.append(os.path.join(root, file))

# letter_signal_paths.sort()
# letter_index_paths.sort()
# reset_signal_paths.sort()
# reset_index_paths.sort()


# # letter signal 
# letter_signal_path = letter_signal_paths[0]
# letter_index_path = letter_index_paths[0]
# letter_signal = np.load(letter_signal_path)
# letter_index = np.load(letter_index_path)
# letter = []                                          # 同一轮的数据保存为一个list，每个元素为一个ndarray，从A到Z;
# for idx in letter_index:
#     letter.append(letter_signal[:, idx[0]:idx[1]])

# # reset signal
# # each circle of reset action name is [UD, MU, MD, DU]
# reset_signal_path = reset_signal_paths[0]
# reset_index_path = reset_index_paths[0]
# reset_signal = np.load(reset_signal_path)
# reset_index = np.load(reset_index_path)
# reset = []
# for idx in reset_index:
#     reset.append(reset_signal[:, idx[0]:idx[1]])


# letterA = letter[0]
# letterM = letter[12]
# reset = reset[1]

# def merge(mat1:np.ndarray, mat2:np.ndarray)-> np.ndarray:
#     if mat1.shape[1] == 0:
#         return mat1
#     mat = np.zeros_like(mat1)
#     for i in range(mat.shape[0]):
#         for j in range(mat.shape[1]):
#             mat[i][j] = max(mat1[i][j], mat2[i][j])
#     return mat

# # overlap_cols_pre = 5
# # overlap_cols_last = int(0.3*reset.shape[1])

# # w_1 = letterA
# # w_2 = np.zeros(shape=(114, abs(overlap_cols_pre)), dtype=letterA.dtype)
# # reset_1 = np.zeros_like(w_2)
# # reset_2 = reset[:, :-overlap_cols_last]
# # reset_3 = reset[:, -overlap_cols_last:]

# # next_w_1 = letterM[:, :overlap_cols_last]
# # next_w_2 = letterM[:, overlap_cols_last:]

# # overlap_mat_1 = merge(w_2, reset_1)
# # overlap_mat_2 = merge(reset_3, next_w_1)

# # word = np.concatenate((w_1, overlap_mat_1, reset_2, overlap_mat_2, next_w_2), axis=1)




# f = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/figure3-5f.npy")
# t = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/figure3-5t.npy")

# # plt.figure(figsize=(12, 8))
# # plt.set_cmap("parula")
# # plt.pcolormesh(t[:word.shape[1]], f, word)
# # plt.ylim([18700, 19300])
# # plt.gca().xaxis.set_major_locator(plt.NullLocator())
# # plt.gca().yaxis.set_major_locator(plt.NullLocator())
# # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# # plt.margins(0, 0)
# # plt.savefig("./img/paper/figure3-10_a.jpg", dpi=600)
# # plt.close()

# overlap_cols_pre = 10
# overlap_cols_last = int(0.3*reset.shape[1])

# w_1 = letterA[:, :-overlap_cols_pre]
# w_2 = letterA[:, -overlap_cols_pre:]

# reset_1 = reset[:, :overlap_cols_pre]
# reset_2 = reset[:, overlap_cols_pre:-overlap_cols_last]
# reset_3 = reset[:, -overlap_cols_last:] 

# next_w_1 = letterM[:, :overlap_cols_last]
# next_w_2 = letterM[:, overlap_cols_last:]

# overlap_mat_1 = merge(w_2, reset_1)
# overlap_mat_2 = merge(reset_3, next_w_1)

# word = np.concatenate((w_1, overlap_mat_1, reset_2, overlap_mat_2, next_w_2), axis=1)


# plt.figure(figsize=(5, 8))
# plt.set_cmap("parula")
# plt.pcolormesh(t[:letterA.shape[1]], f, letterA)
# plt.ylim([18700, 19300])
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.savefig("./img/paper/ppt_letterA.jpg", dpi=600)
# plt.close()

# plt.figure(figsize=(5, 8))
# plt.set_cmap("parula")
# plt.pcolormesh(t[:letterM.shape[1]], f, letterM)
# plt.ylim([18700, 19300])
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.savefig("./img/paper/ppt_letterM.jpg", dpi=600)
# plt.close()

# plt.figure(figsize=(2, 8))
# plt.set_cmap("parula")
# plt.pcolormesh(t[:reset.shape[1]], f, reset)
# plt.ylim([18700, 19300])
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.savefig("./img/paper/ppt_reset.jpg", dpi=600)
# plt.close()

# plt.figure(figsize=(12, 8))
# plt.set_cmap("parula")
# plt.pcolormesh(t[:word.shape[1]], f, word)
# plt.ylim([18700, 19300])
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.savefig("./img/paper/ppt_am.jpg", dpi=600)
# plt.close()

# exit(0)