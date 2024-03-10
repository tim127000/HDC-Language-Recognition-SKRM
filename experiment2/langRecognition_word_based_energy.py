import numpy as np
import os
from tqdm import tqdm
import skrm_word_based_energy as skrm
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description = 'set up your skrymion racetrack memory')
    parser.add_argument('-s' , '--segment_size' , default = 64 , type = int)
    parser.add_argument('-t' , '--track_size' , default = 1152 , type = int)
    parser.add_argument('-d' , '--DBC_capacity' , default = 3 , type = int)
    parser.add_argument('-v' , '--HV_size' , default = 8192 , type = int)
    return parser

def shiftAll(track_size , segment_size , n , tile):
    track_capacity = track_size // segment_size
    detect = 8
    insert = 0
    shift = 0
    remove = 0
    for dbc in tile:
        dbc.shiftR(1 , track_capacity) #shift from the AP of the first segment to the end of the track
    for dbc_id , dbc in enumerate(tile):
        for track_id in range(n):
            if dbc.racetrack[track_id][(track_capacity-1) * segment_size] == 1:
                insert = insert + 1
            tile[(dbc_id+1) % len(tile)].racetrack[track_id][segment_size] = tile[dbc_id].racetrack[track_id][(track_capacity-1) * segment_size]
    skrm.updateCount(shift , detect , insert , remove)

def genRandomHV(HV_size):
    if HV_size%2 != 0:
        print("Dimension is odd!!")
    else:
        randomIndex = np.arange(HV_size)
        randomHV = np.zeros(HV_size , dtype = int)
        np.random.shuffle(randomIndex)
        
        for i in randomIndex[0:int(HV_size/2)]:
            randomHV[i] = 1
        for i in randomIndex[int(HV_size/2):]:
            randomHV[i] = 0
        return randomHV

def lookupItemMemory(itemMemory , key , HV_size):
    if key in itemMemory:
        randomHV = itemMemory[key];
    else:
        itemMemory[key] = genRandomHV(HV_size)
        randomHV = itemMemory[key];
    return randomHV

def hammingDistance(u , v , HV_size):
    sum = 0
    temp = u ^ v
    for i in range(HV_size):
        if temp[i] == 0:
            sum = sum + 1
    return sum / HV_size

nGrams = np.zeros(8192 , dtype = int)

def computeSumHV(buffer , itemMemory , tile , args):
    
    global nGrams
    track_size = args.track_size
    segment_size = args.segment_size
    n = args.DBC_capacity
    HV_size = args.HV_size
    real_track_size = track_size - segment_size*2
    count = 0
    tmp = np.zeros(real_track_size , dtype = int)
    sumHV = np.zeros(HV_size , dtype = int)
    shift = 0
    detect = 0
    insert = 0
    remove = 0
    
    for index in tqdm(range(len(buffer))):
        letter = buffer[index]

        #shift then write
        shiftAll(track_size , segment_size , n , tile)
        data = lookupItemMemory(itemMemory , letter , HV_size)
        for i , dbc in enumerate(tile):
            dbc.write(index%3 , data[1024*i:1024*(i+1)])

        #calculate trigramHV and textHV
        #skip the skrm operation, only calculate the operation count
        if index >= n:
            for i , dbc in enumerate(tile):
                shift = shift + segment_size*4
                detect = detect + segment_size
                """
                for j in range(real_track_size):
                    if nGrams[1024*i+j] == 1 and dbc.racetrack[0][64+j] ^ dbc.racetrack[1][64+j] ^ dbc.racetrack[2][64+j] == 0:
                        remove = remove + 1
                    if nGrams[1024*i+j] == 0 and dbc.racetrack[0][64+j] ^ dbc.racetrack[1][64+j] ^ dbc.racetrack[2][64+j] == 1:
                        insert = insert + 1
                """
                tmp = dbc.racetrack[0][segment_size:segment_size+real_track_size] ^ dbc.racetrack[1][segment_size:segment_size+real_track_size] ^ dbc.racetrack[2][segment_size:segment_size+real_track_size]
                for j in range(real_track_size):
                    if tmp[j] == 1 and nGrams[1024*i+j] == 0:
                        insert = insert + 1
                    elif tmp[j] == 0 and nGrams[1024*i+j] == 1:
                        remove = remove + 1
                nGrams[1024*i:1024*(i+1)] = tmp
            sumHV = sumHV + nGrams
            count = count + 1

    skrm.updateCount(shift , detect , insert , remove)
        
    for i in range(len(sumHV)):
        if sumHV[i] > count / 2:
            sumHV[i] = 1
        else:
            sumHV[i] = 0

    return sumHV

def buildLanguageHV(langAM , tile , args):
    iM = {}
    langLabels = ['afr', 'bul', 'ces', 'dan', 'nld', 'deu', 'eng', 'est', 'fin', 'fra', 'ell', 'hun', 'ita', 'lav', 'lit', 'pol', 'por', 'ron', 'slk', 'slv', 'spa', 'swe']
    for i in range(len(langLabels)):
        fileAddress = '../training_texts/'+langLabels[i]+'.txt'
        with open(fileAddress , 'r') as fp:
            buffer = fp.read()
        print(f"start computing {langLabels[i]}.txt...")
        langHV = computeSumHV(buffer[1:10000] , iM , tile , args)
        langAM[langLabels[i]] = langHV
    return iM

def test(iM , langAM , tile , args):
    total = 0
    correct = 0
    textHV = np.zeros(args.DBC_capacity)
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

    for filename in tqdm(os.listdir('../testing_texts')):
        if filename[-3:] == 'txt' and int(filename[3]) < 4:
            actualLabel = filename[0:2]
            with open(os.path.join('../testing_texts' , filename) , 'r') as fp:
                buffer = fp.read()
            print(f'start computing test data {filename}...')
            textHV = computeSumHV(buffer , iM , tile , args)
            maxDistance = -1
            for i in tqdm(range(len(langLabels))):
                distance = hammingDistance(langAM[langLabels[i]] , textHV , args.HV_size)
                if distance > maxDistance:
                    maxDistance = distance
                    predictLang = langLabels[i]
            if predictLang == langMap[actualLabel]:
                correct += 1
            total += 1
    accuracy = correct / total
    skrm.printResult(accuracy , args)

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    dbc1 = skrm.DBC(args)
    dbc2 = skrm.DBC(args)
    dbc3 = skrm.DBC(args)
    dbc4 = skrm.DBC(args)
    dbc5 = skrm.DBC(args)
    dbc6 = skrm.DBC(args)
    dbc7 = skrm.DBC(args)
    dbc8 = skrm.DBC(args)

    tile = []
    tile.append(dbc1)
    tile.append(dbc2)
    tile.append(dbc3)
    tile.append(dbc4)
    tile.append(dbc5)
    tile.append(dbc6)
    tile.append(dbc7)
    tile.append(dbc8)

    langAM = {}
    iM = buildLanguageHV(langAM , tile , args)
    test(iM , langAM , tile , args) 
