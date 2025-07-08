from sklearn.metrics import label_ranking_loss
import torch
import torch.nn as nn
from torch.utils.data import Dataset,TensorDataset
import os
import numpy as np
import TrainParam

class MyDataset(Dataset):
    def __init__(self, signals, labels):
        super(MyDataset, self).__init__()
        # 输入的 signals 为单词的list，其中每个元素为一个单独的单词   [batch, input_size, channel, h_x, w_x]
        # labels中的元素为对应的signals中的元素的标签                [batch, input_size]
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        current_signal = self.signals[index]
        current_label = self.labels[index]
        
        return current_signal, current_label

def MyCollate(batch):
    # 根据signal的长度进行从大到小排序
    batch.sort(key=lambda x:len(x[0]), reverse=True)

    
    datas = [torch.tensor(temp[0]).float() for temp in batch]
    labels = [torch.tensor(temp[1]).long() for temp in batch]

    # 记录每一个data的长度和label的长度
    data_lengths = [len(data) for data in datas]
    label_lengths = [len(label) for label in labels]

    # padding
    datas = nn.utils.rnn.pad_sequence(datas, batch_first=True, padding_value=0.0)
    data_lengths = torch.tensor(data_lengths, dtype=torch.int64)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=TrainParam.CHAR2INT['PAD'])
    label_lengths = torch.tensor(label_lengths, dtype=torch.int64)
    # 返回signal label 和 各自的length
    return datas, labels, data_lengths, label_lengths

def GetData(root_path):
    """
    根据指定路径 root_path 读取原始数据，将原始数据以及对应的label和userid回传

    Parameters
    ----------
        root_path:

    """
    words = []
    labels = []
    ids = []
    paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            pt = os.path.join(root, file)
            if pt.endswith(".npy"):
                paths.append(pt)

    paths.sort()
    for pt in paths:
        words.append(np.load(pt))
        labels.append((pt.split("/")[-1]).split("_")[0])
        ids.append((pt.split("/")[-1]).split("_")[-1][:-4])
    
    return words, labels, ids

class MyDataset_test(Dataset):
    def __init__(self, signals, labels, ids):
        super(MyDataset_test, self).__init__()
        # 输入的 signals 为单词的list，其中每个元素为一个单独的单词   [batch, input_size, channel, h_x, w_x]
        # labels中的元素为对应的signals中的元素的标签                [batch, input_size]
        self.signals = signals
        self.labels = labels
        self.ids = ids

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        current_signal = self.signals[index]
        current_label = self.labels[index]
        current_id = self.ids[index]
        
        return current_signal, current_label, current_id

def MyCollate_test(batch):
    # 根据signal的长度进行从大到小排序
    batch.sort(key=lambda x:len(x[0]), reverse=True)

    
    datas = [torch.tensor(temp[0]).float() for temp in batch]
    labels = [torch.tensor(temp[1]).long() for temp in batch]
    ids = [temp[2] for temp in batch]
    ids = torch.tensor(ids, dtype=torch.int64)

    # 记录每一个data的长度和label的长度
    data_lengths = [len(data) for data in datas]
    label_lengths = [len(label) for label in labels]

    # padding
    datas = nn.utils.rnn.pad_sequence(datas, batch_first=True, padding_value=0.0)
    data_lengths = torch.tensor(data_lengths, dtype=torch.int64)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=TrainParam.CHAR2INT['PAD'])
    label_lengths = torch.tensor(label_lengths, dtype=torch.int64)
    # 返回signal label 和 各自的length
    return datas, labels, data_lengths, label_lengths, ids




class PairedData(object):
    # 这个类直接沿用就行了，不需要更改
    def __init__(self, data_loader_A, data_loader_B, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        A, A_labels = None, None
        B, B_labels = None, None
        try:
            A, A_labels, A_lengths, A_label_lengths = next(self.data_loader_A_iter)
            # A, A_paths, A_indexes = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A_labels is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A_labels, A_lengths, A_label_lengths = next(self.data_loader_A_iter)

        try:
            B, B_labels, B_lengths, B_label_lengths = next(self.data_loader_B_iter)
            # B, B_paths, B_indexes = next(self.data_loader_B_iter)
        except StopIteration:
            if B is None or B_labels is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_labels,B_lengths, B_label_lengths = next(self.data_loader_B_iter)

        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'data_src': A, 'label_src': A_labels, 'data_length_src': A_lengths, 'label_length_src':A_label_lengths,
                    'data_trg': B, 'label_trg': B_labels, 'data_length_trg': B_lengths, 'label_length_trg':B_label_lengths}

class UnalignedDataLoader():
    def __init__(self, batch_size=32, load_thread_num=4):
        super(UnalignedDataLoader, self).__init__()
        self.batch_size = batch_size
        self.load_thread_num = load_thread_num

    def initialize(self, source_signals, source_labels, target_signals, target_labels):
        dataset_source = MyDataset(source_signals, source_labels)
        dataset_target = MyDataset(target_signals, target_labels)
        data_loader_s = torch.utils.data.DataLoader(
            dataset=dataset_source,
            batch_size = self.batch_size,
            shuffle=True,
            num_workers=self.load_thread_num,
            collate_fn=MyCollate
            )

        data_loader_t = torch.utils.data.DataLoader(
            dataset=dataset_target,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.load_thread_num,
            collate_fn=MyCollate
            )
        self.dataset_s = dataset_source
        self.dataset_t = dataset_target
        self.paired_data = PairedData(data_loader_s, data_loader_t,
                                      float("inf"))

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_s), len(self.dataset_t)), float("inf"))
    