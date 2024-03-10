import numpy as np
from scipy.ndimage import shift
from datetime import datetime

shiftCount = 0
detectCount = 0
insertCount = 0
removeCount = 0

class DBC:
    
    #track size : 1024 bits (overhead region 128 bits)
    #track number : 3
    #data segment size : 64bits
    #access port number : 17
    #access port index : 0 ~ 18 (0, 18 represent head and tail of the track)
    #HDC vector size : 8192bits
    
    def __init__(self , args):
        self.racetrack = np.zeros((args.DBC_capacity , args.track_size) , dtype = int)
        self.segment_size = args.segment_size
        self.DBC_capacity = args.DBC_capacity
        self.track_size = args.track_size
        self.HV_size = args.HV_size

    def ap2index(self , ap):
        if ap == self.track_size // self.segment_size:
            return ap * self.segment_size - 1
        else:
            return ap * self.segment_size

    def detect(self , ap):
        global detectCount
        data = np.zeros(self.DBC_capacity , dtype = int)
        detectCount = detectCount + 1
        for i in range(self.DBC_capacity):
            data[i] = self.racetrack[i][self.ap2index(ap)]
        return data

    def shiftR(self , ap1 , ap2):
        global shiftCount
        ap1 = self.ap2index(ap1)
        ap2 = self.ap2index(ap2)
        for i in range(self.DBC_capacity):
            self.racetrack[i][ap1:ap2] = shift(self.racetrack[i][ap1:ap2] , 1 , cval = 0)
        shiftCount = shiftCount + 1
    
    def shiftL(self , ap1 , ap2):
        global shiftCount
        ap1 = self.ap2index(ap1)
        ap2 = self.ap2index(ap2)
        for i in range(self.DBC_capacity):
            self.racetrack[i][ap1:ap2] = shift(self.racetrack[i][ap1:ap2] , -1 , cval = 0)
        shiftCount = shiftCount + 1
 
    def write(self , block , data):
        global shiftCount
        global insertCount
        global detectCount
        global removeCount
        real_track_size = self.track_size - self.segment_size * 2
        data_in_racetrack = self.HV_size // self.DBC_capacity

        shiftCount = shiftCount + self.segment_size * 2 * self.DBC_capacity
        detectCount = detectCount + self.segment_size * self.DBC_capacity
        
        for i in range(data_in_racetrack):  #256
            for j in range(self.DBC_capacity):   #32
                if self.racetrack[j][self.segment_size + block * data_in_racetrack + i] == 1 and data[data_in_racetrack * j + i] == 0:
                    removeCount = removeCount + 1
                elif self.racetrack[j][self.segment_size + block * data_in_racetrack + i] == 0 and data[data_in_racetrack * j + i] == 1:
                    insertCount = insertCount + 1
                self.racetrack[j][self.segment_size + block * data_in_racetrack + i] = data[data_in_racetrack * j + i]
"""
    def insert(self , rt , ap):
        global insertCount
        self.racetrack[rt][self.ap2index(ap)] = 1
        insertCount = insertCount + 1

    def remove(self , rt , ap):
        global removeCount
        self.racetrack[rt][self.ap2index(ap)] = 0
        removeCount = removeCount + 1
"""


def compare(randomHV , racetrack):
    for block in range(block_count):
        same = 0
        for i in range(data_in_racetrack):
            for j in range(DBC_capacity):
                if racetrack[j][segment_size + block * 256 + i] == randomHV[DBC_capacity * i + j]:
                    same = same + 1
        if same == HDC_vector_size:
            print("same")
        else:
            print("difference")
    print("--------")


def updateCount(shift , detect , insert , remove):
    
    global shiftCount
    global insertCount
    global removeCount
    global detectCount

    shiftCount = shiftCount + shift
    detectCount = detectCount + detect
    insertCount = insertCount + insert
    removeCount = removeCount + remove

def printResult(accuracy , args):
    
    global shiftCount
    global insertCount
    global removeCount
    global detectCount
    
    print("set up")
    print(f"{args.n_gram}-gram")
    print(f"segment size : {args.segment_size}")
    print(f"track_size : {args.track_size}")
    print(f"DBC_capacity : {args.DBC_capacity}")
    print("")
    print("operation count")
    print(f"insert count : {insertCount}")
    print(f"shift count : {shiftCount}")
    print(f"detect count : {detectCount}")
    print(f"remove count : {removeCount}")
    print(f"The accuracy is {accuracy}")
    
    now = datetime.now()
    month = now.strftime("%m")
    day = now.strftime("%d")
    
    with open(f'bit_interleaved_{month}{day}' , 'w+') as f:
        f.write("experiment set up\n")
        f.write(f"track_size : {args.track_size}\n")
        f.write(f"segment_size : {args.segment_size}\n")
        f.write(f"DBC_capacity : {args.DBC_capacity}\n")
        f.write(f"HDC_vector_size : {args.HV_size}\n\n")

        f.write("experiment result\n")

        f.write(f"insert count : {insertCount}\n")
        f.write(f"shift count : {shiftCount}\n")
        f.write(f"detect count : {detectCount}\n")
        f.write(f"remove count : {removeCount}\n")
        f.write(f"The accuracy is {accuracy}\n")
