# import numpy as np
import os
import random

dictionarys = [
    'AM','ARE','BUT','CAN','DO',\
    'FAMILY','GIVE','HELLO','HOW','IS',\
    'JOB','MAKE','NEXT', 'PERSON', 'PUT',\
    'QUIT','SAY','THAT','THE','WHAT',\
    'WHERE','WHY','WORLD','YOU','ZERO']


_int2letter = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G',\
    7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N',\
    14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T',\
    20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'
}

def RandomFactor(Low:int = 0, High:int = 25, flag:str='N', mean:int = 1, sigma:int = 2) -> int:
    '''
        @Args
        -----
            Low(int): the min of distribute
            High(int): the max of distribute
            flag(str): select which distribute will be use. ['N', 'U']
            mean(int): A options. when flag is N. this arg will be use
            sigma(int): A options. same as mean
    
        @Return
        -------
            nResult(int): random number.
    '''
    if flag == 'N':
        nResult = int(random.gauss(mean, sigma))
        if nResult < Low:
            nResult = Low
        elif nResult > High:
            nResult = High
        return nResult

    if flag == 'U':
        nResult = random.randint(Low, High)
        return nResult


readerFd = open("./resource/Grams_v6.txt", mode="a")


Bigrams = {}

for word in dictionarys: 
    if len(word) == 2:          # 改  2-gram
        for i in range(len(word)):
            change_ch = _int2letter[RandomFactor(0, 25, flag='U')]

            while(change_ch == word[i]):
                change_ch = _int2letter[RandomFactor(0, 25, flag='U')]

            Bigram = word[0:i] + change_ch + word[i+1:]
            if Bigram in Bigrams:
                Bigrams[Bigram] += 1
            else:
                Bigrams[Bigram] = 1

    if len(word) > 2:         # 删   2-gram
        for i in range(len(word) - 1):
            Bigram = word[i:i+2]
            if Bigram in dictionarys:           # 若长度为2的字符组合在dictionary中出现过，则替换其中某个字符
                continue

            if Bigram in Bigrams:
                Bigrams[Bigram] += 1
            else:
                Bigrams[Bigram] = 1

Bigrams = sorted(Bigrams.items(), key=lambda x: x[1], reverse=True)
# print(len(Bigrams))
Bigrams = dict(Bigrams)
# print(Bigrams)
for gram, times in Bigrams.items():
    readerFd.write(gram)
    readerFd.write("\n")

# exit(0)
Trigrams = {}

for word in dictionarys:
    if len(word) == 2:                 # 增
        for i in range(len(word) + 1):
            ch_index = RandomFactor(0, 25, flag='U')
            ch = _int2letter[ch_index]
            Trigram = word[:i] + ch + word[i:]

            if Trigram in Trigrams:
                Trigrams[Trigram] += 1
            else:
                Trigrams[Trigram] = 1

    elif len(word) > 3:                 # 删
        for i in range(len(word) - 2):
            Trigram = word[i:i+3]
            if Trigram in dictionarys:  # 若长度为3的字符组合在dictionary中出现过，则随机替换其中一个字符
                continue

            if Trigram in Trigrams:
                Trigrams[Trigram] += 1
            else:
                Trigrams[Trigram] = 1

Trigrams = sorted(Trigrams.items(), key=lambda x: x[1], reverse=True)
# print(len(Trigrams))
Trigrams = dict(Trigrams)
# print(Trigrams)

for gram, times in Trigrams.items():
    readerFd.write(gram)
    readerFd.write("\n")

# # exit(0)
# bigrams_list = []
# for gram, times in Bigrams.items():
#     bigrams_list.append(gram)

# trigrams_list = []
# for gram, items in Trigrams.items():
#     trigrams_list.append(gram)


# Quadrigrams = {}
# random.shuffle(bigrams_list)

# for i in range(len(bigrams_list)):
#     flag = RandomFactor(0, 1, flag='U')
#     index = RandomFactor(0, len(bigrams_list)-1, flag='U')
#     while i == index:
#         index = RandomFactor(0, len(bigrams_list)-1, flag='U')
    
#     bigram1 = bigrams_list[i]
#     bigram2 = bigrams_list[index]
#     if flag == 0:
#         Quadrigram = bigram1 + bigram2
#     elif flag == 1:
#         Quadrigram = bigram2 + bigram1
    
#     if Quadrigram in Quadrigrams:
#         Quadrigrams[Quadrigram] += 1
#     else:
#         Quadrigrams[Quadrigram] = 1

# Quadrigrams = sorted(Quadrigrams.items(), key=lambda x: x[1], reverse=True)
# # print(len(Quadrigrams))
# Quadrigrams = dict(Quadrigrams)
# # print(Quadrigrams)
# for gram, times in Quadrigrams.items():
#     readerFd.write(gram)
#     readerFd.write("\n")

# Quintugrams = {}

# random.shuffle(bigrams_list)
# random.shuffle(trigrams_list)

# for i in range(25):
#     flag = RandomFactor(0, 1, flag='U')
#     bigram = bigrams_list[i]
#     trigram = trigrams_list[i]
#     if flag == 0:
#         Quintugram = bigram + trigram
#     elif flag == 1:
#         Quintugram = trigram + bigram
    
#     if Quintugram in Quintugrams:
#         Quintugrams[Quintugram] += 1
#     else:
#         Quintugrams[Quintugram] = 1


# Quintugrams = sorted(Quintugrams.items(), key=lambda x: x[1], reverse=True)
# # print(len(Quintugrams))
# Quintugrams = dict(Quintugrams)
# # print(Quintugrams)
# for gram, times in Quintugrams.items():
#     readerFd.write(gram)
#     readerFd.write("\n")

# # exit(0)
# Sextugrams = {}
# # random.shuffle(bigrams_list)
# random.shuffle(trigrams_list)

# for i in range(25):
#     flag = RandomFactor(0, 1, flag='U')
    
#     trigram1 = trigrams_list[i]
#     trigram2 = trigrams_list[-(i+1)]
#     if flag == 0:
#         Sextugram = trigram1 + trigram2
#     elif flag == 1:
#         Sextugram = trigram2 + trigram1
    
#     if Sextugram in Sextugrams:
#         Sextugrams[Sextugram] += 1
#     else:
#         Sextugrams[Sextugram] = 1


# Sextugrams = sorted(Sextugrams.items(), key=lambda x: x[1], reverse=True)
# # print(len(Sextugrams))
# Sextugrams = dict(Sextugrams)
# # print(Sextugrams)
# for gram, times in Sextugrams.items():
#     readerFd.write(gram)
#     readerFd.write("\n")

readerFd.close()