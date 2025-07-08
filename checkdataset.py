import numpy as np
import matplotlib.pyplot as plt
import os

def main1():
    # print("fuck")
    root_path = "../DATA/synthe_dataset_reset_diff_speed_v9"
    save_path = "./img/JS/"

    datas = np.load(os.path.join(root_path, "synthe_test_datas.npy"), allow_pickle=True).tolist()
    labels = np.load(os.path.join(root_path, "synthe_test_labels.npy"), allow_pickle=True).tolist()
    # print(len(datas))

    _int2letter = {
        0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G',\
        7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N',\
        14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T',\
        20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'
    }

    def int2char(label):
        charlabel = ""
        for l in label:
            charlabel += _int2letter[l]
        return charlabel


    for i in range(0, len(datas), 25):
        data = datas[i]
        label = int2char(labels[i])
        
        pth = os.path.join(save_path, label + "_" +str(i))
        if os.path.exists(pth) is False:
            os.mkdir(pth)
        
        for j in range(len(data)):
            d = data[j][0]
            plt.figure()
            plt.imshow(d)
            plt.savefig(os.path.join(pth, str(j) + ".png"))
            plt.close()
        # exit(0)

# main1()


def main2():
    root_path = "/sdo/zcl/AcouWrite/DATA/ALL/final_real_word/WDY_WORD_bu"
    # root_path = "/sdo/zcl/AcouWrite/DATA/forpaper/shuoming/data"
    save_path = "./img/WDY"
    for root, dir, files in os.walk(root_path):
        for file in files:
            if file.endswith(".txt"):
                continue
            pth = os.path.join(root, file)
            data = np.load(pth)
            label = file.split("_")[0]
            # print(label)
            spth = os.path.join(save_path, label)
            if os.path.exists(spth) is False:
                os.mkdir(spth)
            
            plt.figure(figsize=(8,6))
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.imshow(np.flip(data, axis=0))
            plt.savefig(os.path.join(spth, file[:-4] + ".png"))
            plt.close()

# print("fuck")
main2()