import numpy as np
import os
from tqdm import tqdm
import skrm_bit_interleaved as skrm
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description = 'set up your skyrmion racetrack memory')
    parser.add_argument('-s' , '--segment_size' , default = 64 , type = int)
    parser.add_argument('-t' , '--track_size' , default = 1152 , type = int)
    parser.add_argument('-d' , '--DBC_capacity' , default = 32 , type = int)
    parser.add_argument('-v' , '--HV_size' , default = 8192 , type = int)
    parser.add_argument('-n' , '--n_gram' , default = 3 , type = int)
    return parser

def shiftAll(dbc , args):
    DBC_capacity = args.DBC_capacity
    track_size = args.track_size
    segment_size = args.segment_size
    HV_size = args.HV_size

    real_track_size = track_size - segment_size * 2
    data_in_racetrack = HV_size // dbc.DBC_capacity
    block_count = track_size // data_in_racetrack
    detect = block_count
    insert = block_count
    shift = 1
    remove = 0
    dbc.shiftR(1 , 18)
    for block in range(block_count):
        for track in range(DBC_capacity):
            if track+1 >= dbc.DBC_capacity:
                dbc.racetrack[0][block * data_in_racetrack] = dbc.racetrack[track][(block+1) * data_in_racetrack]
            else:
                dbc.racetrack[track+1][block * data_in_racetrack] = dbc.racetrack[track][(block+1) * data_in_racetrack]
    skrm.updateCount(shift , detect , insert , remove)

def genRandomHV(HV_size):
    if HV_size%2 != 0:
        print("Dimension is odd!!")
    else:
        randomIndex = np.arange(HV_size)
        randomHV = np.zeros(HV_size , dtype = int)
        np.random.shuffle(randomIndex)

        for i in randomIndex[0:HV_size//2]:
            randomHV[i] = 1
        for i in randomIndex[HV_size//2:]:
            randomHV[i] = 0
        return randomHV

def lookupItemMemory(itemMemory , key , HV_size):
    if key in itemMemory:
        randomHV = itemMemory[key];
    else:
        itemMemory[key] = genRandomHV(HV_size)
        randomHV = itemMemory[key];
    return randomHV

def HammingDistance(u , v , HV_size):
    sum = 0
    temp = u ^ v
    for i in range(HV_size):
        if temp[i] == 0:
            sum = sum + 1
    return sum / HV_size

def computeSumHV(buffer , itemMemory , dbc , args):
    global nGrams
    track_size = args.track_size
    segment_size = args.segment_size
    HV_size = args.HV_size
    DBC_capacity = args.DBC_capacity
    data_in_racetrack = HV_size // DBC_capacity
    
    sumHV = np.zeros(HV_size , dtype = int)
    nGrams = np.zeros(HV_size , dtype = int)
    count = 0
    shift = 0
    detect = 0
    insert = 0
    remove = 0
    
    for index in tqdm(range(len(buffer))):
        letter = buffer[index]
        
        #shift then write
        shiftAll(dbc , args)
        data = lookupItemMemory(itemMemory , letter , HV_size)
        dbc.write(index % args.n_gram , data)

        #calculate trigramHV and textHV
        #skip the skrm operation, only calculate the operation count
        if index >= args.n_gram:
            shift = shift + segment_size * 4 #dbc and ngrams
            detect = detect + segment_size
            insert = insert + segment_size
            for track in range(DBC_capacity):
                for i in range(data_in_racetrack):
                    nGrams[track * data_in_racetrack + i] = dbc.racetrack[track][segment_size + data_in_racetrack * 0 + i] ^ dbc.racetrack[track][segment_size + data_in_racetrack * 1 + i] ^ dbc.racetrack[track][segment_size + data_in_racetrack * 2 + i]
            sumHV = sumHV + nGrams
            count = count + 1

    skrm.updateCount(shift , detect , insert , remove)
        
    for i in range(len(sumHV)):
        if sumHV[i] > count / 2:
            sumHV[i] = 1
        else:
            sumHV[i] = 0

    return sumHV

def buildLanguageHV(langAM , dbc , args):
    iM = {}
    langLabels = ['afr', 'bul', 'ces', 'dan', 'nld', 'deu', 'eng', 'est', 'fin', 'fra', 'ell', 'hun', 'ita', 'lav', 'lit', 'pol', 'por', 'ron', 'slk', 'slv', 'spa', 'swe']
    for i in range(len(langLabels)):
        fileAddress = './training_texts/'+langLabels[i]+'.txt'
        with open(fileAddress , 'r') as fp:
            buffer = fp.read()
        print(f"start computing {langLabels[i]}.txt...")
        langHV = computeSumHV(buffer[0:10000] , iM , dbc , args)
        langAM[langLabels[i]] = langHV
    return iM

def test(iM , langAM , dbc , args):
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

    for filename in os.listdir('./testing_texts'):
        if filename[-3:] == 'txt' and int(filename[3]) < 4:
            actualLabel = filename[0:2]
            with open(os.path.join('./testing_texts' , filename) , 'r') as fp:
                buffer = fp.read()
            print(f'start computing test data {filename}...')
            textHV = computeSumHV(buffer , iM , dbc , args)
            maxDistance = -1
            for i in range(len(langLabels)):
                distance = HammingDistance(langAM[langLabels[i]] , textHV , args.HV_size)
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
    dbc = skrm.DBC(args)
    
    langAM = {}
    iM = buildLanguageHV(langAM , dbc , args)
    test(iM , langAM , dbc , args)
   

