# import os
# import random

# dictionarys = [
#     'AM','ARE','BUT','CAN','DO',\
#     'FAMILY','GIVE','HELLO','HOW','IS',\
#     'JOB','MAKE','NEXT', 'PERSON', 'PUT',\
#     'QUIT','SAY','THAT','THE','WHAT',\
#     'WHERE','WHY','WORLD','YOU','ZERO']

# _int2letter = {
#     0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G',\
#     7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N',\
#     14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T',\
#     20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'
# }


# def RandomFactor(Low:int = 0, High:int = 25, flag:str='N', mean:int = 1, sigma:int = 2) -> int:
#     '''
#         @Args
#         -----
#             Low(int): the min of distribute
#             High(int): the max of distribute
#             flag(str): select which distribute will be use. ['N', 'U']
#             mean(int): A options. when flag is N. this arg will be use
#             sigma(int): A options. same as mean
    
#         @Return
#         -------
#             nResult(int): random number.
#     '''
#     if flag == 'N':
#         nResult = int(random.gauss(mean, sigma))
#         if nResult < Low:
#             nResult = Low
#         elif nResult > High:
#             nResult = High
#         return nResult

#     if flag == 'U':
#         nResult = random.randint(Low, High)
#         return nResult
    

# readerFd = open("./resource/Grams_v7.txt", mode="a")

# grams = []
# wordlen2 = ['AM','DO','IS']
# for word in wordlen2:
#     for i in range(len(word)):
#         change_ch = _int2letter[RandomFactor(0, 25, flag='U')]

#         while(change_ch == word[i]):
#             change_ch = _int2letter[RandomFactor(0, 25, flag='U')]

#         Bigram = word[0:i] + change_ch + word[i+1:]
#         grams.append(Bigram)
    
#     grams.append(word[0] + word[0] + word[1])
#     grams.append(word[0] + word[1] + word[1])
#     grams.append(_int2letter[RandomFactor(0, 25, flag='U')] + word)
#     grams.append(word[0] + _int2letter[RandomFactor(0, 25, flag='U')] + word[1])
#     grams.append(word + _int2letter[RandomFactor(0, 25, flag='U')])


# grams = list(set(grams))
# # print(grams)
# for gram in grams:
#     readerFd.write(gram)
#     readerFd.write("\n")

# grams.clear()

# wordlen3 = ['ARE','BUT','CAN','HOW','JOB','PUT','SAY','THE','WHY','YOU']
# for word in wordlen3:
#     for i in range(len(word)):
#         change_ch = _int2letter[RandomFactor(0, 25, flag='U')]

#         while(change_ch == word[i]):
#             change_ch = _int2letter[RandomFactor(0, 25, flag='U')]

#         gram = word[0:i] + change_ch + word[i+1:]
#         grams.append(gram)
    
#     grams.append(word[0] + word[0] + word[1] + word[2])
#     grams.append(word[0] + word[1] + word[1] + word[2])
#     grams.append(word[0] + word[1] + word[2] + word[2])
#     # grams.append(_int2letter[RandomFactor(0, 25, flag='U')] + word)
#     grams.append(word[0] + _int2letter[RandomFactor(0, 25, flag='U')] + word[1:])
#     grams.append(word[:2] + _int2letter[RandomFactor(0, 25, flag='U')] + word[2:])
#     grams.append(word + _int2letter[RandomFactor(0, 25, flag='U')])

# for gram in grams:
#     readerFd.write(gram)
#     readerFd.write("\n")

# wordlenmt3 = ['FAMILY','GIVE','HELLO','MAKE','NEXT', 'PERSON', 'QUIT','THAT','WHAT','WHERE','WORLD','ZERO']

# Bigrams = set()
# Trigrams = set()
# for word in wordlenmt3: 
#     for i in range(len(word) - 1):
#         Bigram = word[i:i+2]
#         Bigrams.add(Bigram)

#     for i in range(len(word) - 2):
#         Trigram = word[i:i+3]
#         Trigrams.add(Trigram)

# Bigrams = list(Bigrams)
# Trigrams = list(Trigrams)

# # print(Bigrams)
# # print(Trigrams)

# Quadrigrams = set()
# random.shuffle(Bigrams)

# for i in range(len(Bigrams)):
#     flag = RandomFactor(0, 1, flag='U')
#     index = RandomFactor(0, len(Bigrams)-1, flag='U')
#     while i == index:
#         index = RandomFactor(0, len(Bigrams)-1, flag='U')
    
#     bigram1 = Bigrams[i]
#     bigram2 = Bigrams[index]
#     if flag == 0:
#         Quadrigram = bigram1 + bigram2
#     elif flag == 1:
#         Quadrigram = bigram2 + bigram1
    
#     Quadrigrams.add(Quadrigram)

# # print(Quadrigrams)

# for gram in Quadrigrams:
#     readerFd.write(gram)
#     readerFd.write("\n")

# Quintugrams = set()

# random.shuffle(Bigrams)
# random.shuffle(Trigrams)

# for i in range(25):
#     flag = RandomFactor(0, 1, flag='U')
#     bigram = Bigrams[i]
#     trigram = Trigrams[i]
#     if flag == 0:
#         Quintugram = bigram + trigram
#     elif flag == 1:
#         Quintugram = trigram + bigram
    
#     Quintugrams.add(Quintugram)



# for gram in Quintugrams:
#     readerFd.write(gram)
#     readerFd.write("\n")


# Sextugrams = set()
# random.shuffle(Trigrams)

# for i in range(25):
#     flag = RandomFactor(0, 1, flag='U')
    
#     trigram1 = Trigrams[i]
#     trigram2 = Trigrams[-(i+1)]
#     if flag == 0:
#         Sextugram = trigram1 + trigram2
#     elif flag == 1:
#         Sextugram = trigram2 + trigram1

#     Sextugrams.add(Sextugram)

# for gram in Sextugrams:
#     readerFd.write(gram)
#     readerFd.write("\n")

# readerFd.close()


readerFd = open("./resource/Grams_v5.txt", mode="r")
words = []
for line in readerFd:
    line = line.strip("\n")
    words.append(line)

readerFd.close()
words = sorted(words, key=lambda i: len(i), reverse=False)
print(words[:10])

writeFd = open("./resource/Grams_v7.txt", mode="a")

for word in words:
    writeFd.write(word)
    writeFd.write("\n")

writeFd.close()