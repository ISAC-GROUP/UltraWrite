from asyncore import write
import pandas as pd
import numpy as np
import random

path = "./resource/COCA.xlsx"

test_dictionarys = [
    'AM','ARE','BUT','CAN','DO',\
    'FAMILY','GIVE','HELLO','HOW','IS',\
    'JOB','MAKE','NEXT', 'PERSON', 'PUT',\
    'QUIT','SAY','THAT','THE','WHAT',\
    'WHERE','WHY','WORLD','YOU','ZERO']

readerpd = pd.read_excel(path, usecols=[1])

writerFd = open("./resource/word_v4.txt", mode="a")

words = readerpd.head(n=1000).to_numpy().tolist()
dictionarys = []
for word in words:
    word = word[0]
    if type(word) != str:
        word = str(word)
    word = word.upper()
    dictionarys.append(word)

# print(dictionarys)

random.shuffle(dictionarys)

top150_word = dictionarys[:300]


save_result = []
for word in top150_word:
    if word not in test_dictionarys and len(word) < 7:
        save_result.append(word)
        writerFd.write(word)
        writerFd.write('\n')


# print(save_result)
print(len(save_result))



chardict = {}
for word in save_result:
    for ch in word:
        if ch in chardict:
            chardict[ch] += 1
        else:
            chardict[ch] = 1

chardict = sorted(chardict.items(), key= lambda x:x[0])

print(chardict)
print(len(chardict))

writerFd.close()