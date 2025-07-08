# ================================
# 这个文件保存着各种参数设置
# ===============================

# ======================================================================
# 制作数据集时用参数
# =========================
CHAR2INT = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, \
                "H":7, "I":8, "J":9, "K":10, "L":11, "M":12, "N":13,\
                "O":14, "P":15, "Q":16, "R":17, "S":18, "T":19,\
                "U":20, "V":21, "W":22, "X":23, "Y":24, "Z":25, " ":26,
                "EOS":26, "BOS":27, "PAD":28}
# CHAR2INT = {'A':0, 'D':1, 'E':2, 'H':3, 'I':4, 'L':5, 'M':6, 'N':7, 'O':8,
#                 'R':9, 'S':10, 'T':11, 'U':12, 'W':13, 'Y':14, ' ':15, 
#                 'EOS':15, 'BOS':16, 'PAD':17}

CHARNUM = 26

DATA_PATH = "../DATA/ALL/ZCL_WORD_v3"

NPERSEG = 8192                      # stft 窗口大小
NOVERLAP = 7680                     # 窗口重叠大小
FS = 44100                          # 音频采样率

WINDOW_SIZE = 100                   # 对于单词数据的移动窗口大小 
STRIDE = 30                         # 移动步长


TRAIN_TEST_RATE = 0.8               # 训练测试比例

DATASET_PATH = "../DATA/dataset_26c_25w_100ws_30st_v2"


# ======================================================================
# 训练时使用参数
# =========================
# Dataloader
BATCH_SIZE = 32
NUM_WORKERS = 4

# PL_MODEL
CHARS = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
        'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z']
# CHARS = ['A', 'D', 'E', 'H', 'I', 'L', 'M', 'N', 'O',
#         'R', 'S', 'T', 'U', 'W', 'Y']

SAVE_RESULT = "checkpoint/TXT_Result/synthe2real_diff_speed_v9_2_28.txt"

# SAVE_RESULT = "checkpoint/TXT_Result/cwy_test_v3.txt"
LOG_PATH = "checkpoint"
GPUS = [3]

TRAIN_DATASET_PATH = "../DATA/synthe_dataset_reset_diff_speed_v9"
TEST_DATASET_PATH = "../DATA/final_real_dataset/zpz_dataset_26c_25w_100ws_30st_v2"
# TEST_DATASET_PATH = "../DATA/zpz_dataset_26c_25w_100ws_30st"

TRAINMODE = "B->C"

USEMODEL = "PL_V11"

NET_PARAM = {
    "embed_input_nc":1,
    "embed_output_size":128,
    "embed_input_shape":[114, WINDOW_SIZE],
    "dropout":0.3,

    "input_size":128,
    "hidden_size":512,
    "num_layer":2,
    "directions":True,

    "encoder_output_size":CHARNUM+1,        # include <BLANK>
    "decoder_output_size":CHARNUM+1,        # include <EOS>
}


# NET_PARAM = {
#     "front_end_in_channel":1,
#     "front_end_out_channel":128,
#     "front_end_input_shape":[114, WINDOW_SIZE],
#     "front_end_dropout":0.4,

#     "encoder_out_channel":128,
#     # "encoder_num_layers":20,
#     # "encoder_num_stacks":2,
#     "encoder_dilate_rate":[1,2,5,2,2,5],
#     "encoder_kernel_size":5,
#     "encoder_residual_channels":128,
#     "encoder_gate_channels":128,
#     "encoder_skip_out_channels":128,
#     "encoder_fc_out_size":27,

#     "decoder_input_size":32,
#     "decoder_hidden_size":128,
#     "decoder_num_layer":2,
#     "decoder_direction":False,
#     "decoder_num_embed":28,
#     "decoder_embed_dim":32,
#     "decoder_dropout":0.0,
#     "decoder_output_size":27,
#     "decoder_atten_mode":'general',
#     "decoder_pad":CHAR2INT['PAD'],
#     "BOS_Token":CHAR2INT['BOS'],
#     "EOS_Token":CHAR2INT['EOS']
# }

TRAIN_PARAM = {
    "CHARS":CHARS,
    "CHAR2INT":CHAR2INT,
    "BLANK":CHAR2INT[' '],
    "BOS":CHAR2INT['BOS'],
    'EOS':CHAR2INT['EOS'],
    'PAD':CHAR2INT['PAD'],
    "lr":1e-3,
    "save_result":SAVE_RESULT,
    "max_length":8,
    "batch_size":BATCH_SIZE,
    "gpus":GPUS,
    "lambda":0.9,
    "train_dataset":TRAIN_DATASET_PATH,
    "test_dataset":TEST_DATASET_PATH,
    "trainmode":TRAINMODE
}

# =====================================================================