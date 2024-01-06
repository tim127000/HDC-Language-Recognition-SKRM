import numpy as np
import os
from tqdm import tqdm
import skrm

# word-based without padding zero

dbc1 = skrm.DBC()
dbc2 = skrm.DBC()
dbc3 = skrm.DBC()
dbc4 = skrm.DBC()
dbc5 = skrm.DBC()
dbc6 = skrm.DBC()
dbc7 = skrm.DBC()
dbc8 = skrm.DBC()

tile = []
tile.append(dbc1)
tile.append(dbc2)
tile.append(dbc3)
tile.append(dbc4)
tile.append(dbc5)
tile.append(dbc6)
tile.append(dbc7)
tile.append(dbc8)

def shiftAll():
    for dbc in tile:
        dbc.shiftR(1 , 18)
    for index , dbc in enumerate(tile):
        for i in range(3):
            if dbc.detect(i , 17):
                tile[(index+1)%8].insert(i , 1)
            skrm.updateCount(0 , -2 , 0 , 0) #detect operations in same DBC can operate parallely

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

nGrams = np.zeros(8192 , dtype = int)

def computeSumHV(buffer , itemMemory , n , dimension):
    
    global nGrams
    count = 0
    sumHV = np.zeros(dimension , dtype = int)
    shift = 0
    detect = 0
    insert = 0
    remove = 0

    for index in tqdm(range(len(buffer))):
        letter = buffer[index]
        
        #shift then write
        shiftAll()
        data = lookupItemMemory(itemMemory , letter , dimension)
        for i , dbc in enumerate(tile):
            dbc.write(index%3 , data[1024*i:1024*(i+1)])

        #calculate trigramHV and textHV
        #skip the skrm operation, only calculate the operation count
        if index >= n:
            for i , dbc in enumerate(tile): 
                shift = shift + 64*4
                detect = detect + 1024
                for j in range(1024):
                    if nGrams[1024*i+j] == 1 and dbc.racetrack[0][64+j] ^ dbc.racetrack[1][64+j] ^ dbc.racetrack[2][64+j] == 0:
                        remove = remove + 1
                    if nGrams[1024*i+j] == 0 and dbc.racetrack[0][64+j] ^ dbc.racetrack[1][64+j] ^ dbc.racetrack[2][64+j] == 1:
                        insert = insert + 1
                nGrams[1024*i:1024*(i+1)] = dbc.racetrack[0][64:1088] ^ dbc.racetrack[1][64:1088] ^ dbc.racetrack[2][64:1088]

            sumHV = sumHV + nGrams
            count = count + 1

    skrm.updateCount(shift , detect , insert , remove)
        
    for i in range(len(sumHV)):
        if sumHV[i] > count / 2:
            sumHV[i] = 1
        else:
            sumHV[i] = 0

    return sumHV

def buildLanguageHV(n , dimension , langAM):
    iM = {}
    langLabels = ['afr', 'bul', 'ces', 'dan', 'nld', 'deu', 'eng', 'est', 'fin', 'fra', 'ell', 'hun', 'ita', 'lav', 'lit', 'pol', 'por', 'ron', 'slk', 'slv', 'spa', 'swe']
    for i in range(len(langLabels)):
        fileAddress = './training_texts/'+langLabels[i]+'.txt'
        with open(fileAddress , 'r') as fp:
            buffer = fp.read()
        print(f"start computing {langLabels[i]}.txt...")
        langHV = computeSumHV(buffer[0:10000] , iM , n , dimension)
        langAM[langLabels[i]] = langHV
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

    for filename in tqdm(os.listdir('./testing_texts')):
        if filename[-3:] == 'txt' and int(filename[3]) < 4:
            actualLabel = filename[0:2]
            with open(os.path.join('./testing_texts' , filename) , 'r') as fp:
                buffer = fp.read()
            print(f'start computing test data {filename}...')
            textHV = computeSumHV(buffer , iM , n , dimension)
            maxDistance = -1
            for i in tqdm(range(len(langLabels))):
                distance = hammingDistance(langAM[langLabels[i]] , textHV , dimension)
                if distance > maxDistance:
                    maxDistance = distance
                    predictLang = langLabels[i]
            if predictLang == langMap[actualLabel]:
                correct += 1
            total += 1
    accuracy = correct / total
    skrm.printResult(accuracy)

if __name__=="__main__":
    langAM = {}
    n = 3
    dimension = 8192
    iM = buildLanguageHV(n , dimension , langAM)
    test(iM , langAM , n , dimension)
