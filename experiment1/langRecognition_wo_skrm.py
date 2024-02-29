import numpy as np
import os
from tqdm import tqdm

def genRandomHV(dimension):
    if dimension%2 != 0:
        print("Dimension is odd!!")
    else:
        randomIndex = np.arange(dimension)
        randomHV = np.zeros(dimension , dtype = int)
        np.random.shuffle(randomIndex)
        
        for i in randomIndex[0:int(dimension/2)]:
            randomHV[i] = 1
        for i in randomIndex[int(dimension/2):]:
            randomHV[i] = 0
        return randomHV

def lookupItemMemory(itemMemory , key , dimension):
    if key in itemMemory:
        randomHV = itemMemory[key];
    else:
        itemMemory[key] = genRandomHV(dimension)
        randomHV = itemMemory[key];
    return randomHV

def hammingDistance(u , v , dimension):
    sum = 0
    temp = u ^ v
    for i in range(dimension):
        if temp[i] == 0:
            sum = sum + 1
    return sum / dimension


def computeSumHV(buffer , itemMemory , n , dimension):
    total_1 = 0
    total_0 = 0
    count = 0
    block = np.zeros((n , dimension) , dtype = int)
    sumHV = np.zeros(dimension , dtype = int)

    for index in range(len(buffer)):
        letter = buffer[index]
        
        #shift then write
        block = np.roll(block , (1,1) , axis=(0,1))
        block[0] = lookupItemMemory(itemMemory , letter , dimension)
        
        #calculate trigramHV and textHV
        if index >= n:
            nGrams = block[0]
            for i in range(1,n):
                nGrams[0] = nGrams[0] ^ block[i][0]
            if nGrams[0] == 1:
                total_1 = total_1 + 1
            if nGrams[0] == 0:
                total_0 = total_0 + 1

    return total_0 , total_1

def buildLanguageHV(n , dimension , langAM):
    global max_1
    global max_0
    iM = {}
    langLabels = ['afr', 'bul', 'ces', 'dan', 'nld', 'deu', 'eng', 'est', 'fin', 'fra', 'ell', 'hun', 'ita', 'lav', 'lit', 'pol', 'por', 'ron', 'slk', 'slv', 'spa', 'swe']
    count_0= np.zeros(len(langLabels) , dtype = int)
    count_1 = np.zeros(len(langLabels) , dtype = int)
    for i in range(len(langLabels)):
        fileAddress = '../training_texts/'+langLabels[i]+'.txt'
        with open(fileAddress , 'r') as fp:
            buffer = fp.read();
        total_0 , total_1 = computeSumHV(buffer , iM , n , dimension)
        count_0[i] = total_0
        count_1[i] = total_1

    with open('nums_of_01' , 'w+') as fp:
        fp.write("count of 0: ")
        for i in range(len(langLabels)):
            fp.write(f"{count_0[i]} ")
        fp.write("\ncount of 1: ")
        for i in range(len(langLabels)):
            fp.write(f"{count_1[i]} ")
        fp.write("\n")
    return iM

def test(iM , langAM , n , dimension):
    total = 0
    correct = 0
    textHV = np.zeros(n)
    predictLang = ''
    langLabels = ['afr', 'bul', 'ces', 'dan', 'nld', 'deu', 'eng', 'est', 'fin', 'fra', 'ell', 'hun', 'ita', 'lav', 'lit', 'pol', 'por', 'ron', 'slk', 'slv', 'spa', 'swe']
    langMap = {}
    langMap['af'] = 'afr'
    langMap['bg'] = 'bul'
    langMap['cs'] = 'ces'
    langMap['da'] = 'dan'
    langMap['nl'] = 'nld'
    langMap['de'] = 'deu'
    langMap['en'] = 'eng'
    langMap['et'] = 'est'
    langMap['fi'] = 'fin'
    langMap['fr'] = 'fra'
    langMap['el'] = 'ell'
    langMap['hu'] = 'hun'
    langMap['it'] = 'ita'
    langMap['lv'] = 'lav'
    langMap['lt'] = 'lit'
    langMap['pl'] = 'pol'
    langMap['pt'] = 'por'
    langMap['ro'] = 'ron'
    langMap['sk'] = 'slk'
    langMap['sl'] = 'slv'
    langMap['es'] = 'spa'
    langMap['sv'] = 'swe'

    for filename in os.listdir('../testing_texts'):
        if filename[-3:] != 'txt':
            break
        actualLabel = filename[0:2]
        with open(os.path.join('../testing_texts' , filename) , 'r') as fp:
            buffer = fp.read()
        print(f'start computing test data {filename}...')
        textHV = computeSumHV(buffer , iM , n , dimension)
        maxDistance = -1
        for i in range(len(langLabels)):
            distance = hammingDistance(langAM[langLabels[i]] , textHV , dimension)
            if distance > maxDistance:
                maxDistance = distance
                predictLang = langLabels[i]
        if predictLang == langMap[actualLabel]:
            correct += 1
        total += 1
    accuracy = correct / total
    print(accuracy)
    print("the accuracy is {:.2f}".format(accuracy))

if __name__ == "__main__":
    langAM = {}
    n = 3
    dimension = 10000
    iM = buildLanguageHV(n , dimension , langAM)
    #test(iM , langAM , n , dimension)

