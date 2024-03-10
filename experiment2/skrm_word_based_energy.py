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
    
    def __init__(self , args):
        self.racetrack = np.zeros((args.DBC_capacity , args.track_size) , dtype = int)
        self.segment = args.segment_size
        self.DBC_capacity = args.DBC_capacity
        self.track_size = args.track_size

    def ap2index(self , ap):
        if ap == self.track_size // self.segment:
            return ap * self.segment - 1
        else:
            return ap * self.segment

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
        shiftCount = shiftCount + 3
    
    def shiftL(self , ap1 , ap2):
        global shiftCount
        ap1 = self.ap2index(ap1)
        ap2 = self.ap2index(ap2)
        for i in range(self.DBC_capacity):
            self.racetrack[i][ap1:ap2] = shift(self.racetrack[i][ap1:ap2] , -1 , cval = 0)
        shiftCount = shiftCount + 3

    def update(self , rt , ap , data):
        origin = self.detect(rt , ap)
        if data == 1 and origin == 0:
            self.insert(rt , ap)
        if data == 0 and origin == 1:
            self.remove(rt , ap)

    def write(self , rt , data):
        """
        for i in range(16):
            self.update(rt , i+1 , data[i*self.segment])
        for i in range(1 , self.segment):
            self.shiftL(0 , 18)
            for j in range(16):
                self.update(rt , j+1 , data[j*self.segment+i])
        for i in range(self.segment-1):
            self.shiftR(0 , 18)
        """
        global shiftCount
        global insertCount
        global detectCount
        global removeCount
        shiftCount = shiftCount + self.segment * 2
        detectCount = detectCount + self.segment
        for i in range(len(data)):
            if self.racetrack[rt][self.segment+i] == 1 and data[i] ==0:
                removeCount = removeCount + 1
            if self.racetrack[rt][self.segment+i] == 0 and data[i] ==1:
                insertCount = insertCount + 1
        self.racetrack[rt][self.segment:-self.segment] = data
    
    def insert(self , rt , ap):
        global insertCount
        self.racetrack[rt][self.ap2index(ap)] = 1
        insertCount = insertCount + 1

    def remove(self , rt , ap):
        global removeCount
        self.racetrack[rt][self.ap2index(ap)] = 0
        removeCount = removeCount + 1

def compare(randomHV , racetrack):
    real_track_size = self.track_size - self.segment * 2
    for i in range(self.DBC_capacity):
        same = 0
        for j in range(real_track_size):
            if randomHV[j] == racetrack[i][j+self.segment_size]:
                same = same + 1
        if same == real_track_size:
            print("same")
        else:
            print("difference")

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
    print("experiment set up")
    print(f"track_size : {args.track_size}")
    print(f"segment_size : {args.segment_size}")
    print(f"DBC_capacity : {args.DBC_capacity}")
    print(f"HDC_vector_size : {args.HV_size}")
    print("")
    print("experiment result")
    print(f"insert count : {insertCount}")
    print(f"shift count : {shiftCount}")
    print(f"detect count : {detectCount}")
    print(f"remove count : {removeCount}")
    print(f"The accuracy is {accuracy}")
    
    now = datetime.now()
    month = now.strftime("%m")
    day = now.strftime("%d")
    with open(f'word_based_{month}{day}' , 'w+') as f:
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
