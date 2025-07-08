import numpy as np

from ctc_decoder import best_path, lexicon_search
from ctc_decoder.bk_tree import BKTree


# mat:TxNxC
def decoder(mat, chars, dec_type):
    res = []
    mat = mat.permute(1,0,2).cpu().detach().numpy()
    for n in mat:
        if dec_type == 'best_path':
            res.append(best_path(n, chars))
        elif dec_type == 'lexicon_search':
            # 这里有问题
            bk_tree = BKTree(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            res.append(lexicon_search(n, chars, bk_tree, 4))
    return np.array(res)


