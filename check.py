from utils import util
import numpy as np
import os


def main():
    path = "/sdo/zcl/AcouWrite/DATA/final_real_dataset"

    datas_src = np.load(os.path.join("/sdo/zcl/AcouWrite/DATA/synthe_dataset_reset_diff_speed_v9", "synthe_test_datas.npy"), allow_pickle=True).tolist()
    labels_src = np.load(os.path.join("/sdo/zcl/AcouWrite/DATA/synthe_dataset_reset_diff_speed_v9", "synthe_test_labels.npy"), allow_pickle=True).tolist()

    datas_trg = np.load(os.path.join(path, "zcl_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
    labels_trg = np.load(os.path.join(path, "zcl_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

    trainloader = util.UnalignedDataLoader()
    trainloader.initialize(datas_src, labels_src, datas_trg, labels_trg)
    trainloader = trainloader.load_data()

    for batch_idx, data in enumerate(trainloader):
        data_src = data['data_src']
        label_src = data['label_src']
        data_length_src = data['data_length_src']
        label_length_src = data['label_length_src']

        data_trg = data['data_trg']
        label_trg = data['label_trg']
        data_length_trg = data['data_length_trg']
        label_length_trg = data['label_length_trg']

        print(batch_idx)
    
    print("over")

main()