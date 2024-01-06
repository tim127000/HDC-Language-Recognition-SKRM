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
    
    def __init__(self):
        self.racetrack = np.zeros((3 , 1152) , dtype = int)
        self.segment = 64

    def ap2index(self , ap):
        if ap == 18:
            return 1151
        else:
            return ap * self.segment

    def detect(self , rt , ap):
        global detectCount
        detectCount = detectCount + 1
        return self.racetrack[rt][self.ap2index(ap)]

    def shiftR(self , ap1 , ap2):
        global shiftCount
        ap1 = self.ap2index(ap1)
        ap2 = self.ap2index(ap2)
        for i in range(3):
            self.racetrack[i][ap1:ap2] = shift(self.racetrack[i][ap1:ap2] , 1 , cval = 0)
        shiftCount = shiftCount + 1
    
    def shiftL(self , ap1 , ap2):
        global shiftCount
        ap1 = self.ap2index(ap1)
        ap2 = self.ap2index(ap2)
        for i in range(3):
            self.racetrack[i][ap1:ap2] = shift(self.racetrack[i][ap1:ap2] , -1 , cval = 0)
        shiftCount = shiftCount + 1

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
        shiftCount = shiftCount + 64 * 2
        detectCount = detectCount + 64
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
    for i in range(3):
        same = 0
        for j in range(1024):
            if randomHV[j] == racetrack[i][j+64]:
                same = same + 1
        if same == 1024:
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

def printResult(accuracy):
    
    global shiftCount
    global insertCount
    global removeCount
    global detectCount
    
    print(f"insert count : {insertCount}")
    print(f"shift count : {shiftCount}")
    print(f"detect count : {detectCount}")
    print(f"remove count : {removeCount}")
    print(f"The accuracy is {accuracy}")
    
    now = datetime.now()
    month = now.strftime("%m")
    day = now.strftime("%d")
    with open(f'output_{month}{day}' , 'w+') as f:
        f.write(f"insert count : {insertCount}\n")
        f.write(f"shift count : {shiftCount}\n")
        f.write(f"detect count : {detectCount}\n")
        f.write(f"remove count : {removeCount}\n")
        f.write(f"The accuracy is {accuracy}\n")
