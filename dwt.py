from __future__ import unicode_literals

import math
import random
import collections

import matplotlib.pyplot as plt
import numpy as np


def randExp(N=1):
    return -np.log(np.random.uniform(0., 1., N))[0]


def HomogenousIRFI(y, k):
    return y / k


def HomogenousPoissonProcess(T, k):
    t = 0
    ts = []
    yc = 0
    while t < T:
        yc += randExp()
        t = HomogenousIRFI(yc, k)
        # ts.extend(t)
        ts.append(t)
    return ts[:-1]

global intensities
intensities = [5, 10, 20, 15, 25, 30, 40, 50]
def NonHomogenousPoissonProcess(T):
    t = 0
    ts = []
    yc = 0
    global intensities 
    random.shuffle(intensities)
    
    for j in range(8):
        yc = 0
        t = 0
        while t < (T/8):
        # print('t', t, type(t))
        # print('T', T, type(T))
            k = intensities[j]
            yc += randExp()
            t = HomogenousIRFI(yc, k) 
        # ts.extend(t)
            ts.append(t + (T/8)*j)
    return ts[:-1]


def NonHomo(T):
    t = 0
    ts = []
    yc = 0
    global intensities
    for j in range(8):
        yc = 0
        t = 0
        while t < (T/8):
        # print('t', t, type(t))
        # print('T', T, type(T))
            k = intensities[j]
            yc += randExp()
            t = HomogenousIRFI(yc, k) 
        # ts.extend(t)
            ts.append(t + (T/8)*j)
    return ts[:-1]


def plotPoissonProcess(pp, T):
    tt = np.linspace(0, T, 1001)
    pps = 0 * tt
    for t in pp:
        pps[tt > t] += 1
    plt.plot(tt, pps)
    plt.xlabel('time axis')
    plt.ylabel('event counter')
    plt.show()


def DTWDistance(s1,s2,w):
    DTW={}
    #w = max(w, abs(len(s1)-len(s2)))
    w = 100
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return math.sqrt(DTW[len(s1)-1, len(s2)-1])

def value_chunkwize(win_size, iterable):
    win_size = win_size * 1
    chunks = [list()]
    for x in iterable:
        i = int(x / win_size)
        if i >= len(chunks):
            for k in range(i - len(chunks)+1):
                chunks.append(list())
        chunks[i].append(x)
    return chunks


def ave(nums):
    if not nums:
        return 0.
    return sum(nums) / len(nums)

def CountEventNum(ts,winsize):
    counter = collections.Counter()
    for t in ts:
        n = int(t / winsize)
        counter[n] += 1
    counts = []
    for i in range(max(counter) + 1):
        counts.append(counter[i])
#     print(counts)
    
    return counts


def run():
    T = 8
    ppn = NonHomogenousPoissonProcess(T)
#     plotPoissonProcess(ppn, T)
    
    ppn1 = NonHomo(T)
#     plotPoissonProcess(ppn1, T)
    

    pph = HomogenousPoissonProcess(T, 40)
#     plotPoissonProcess(pph, T)
    
    count = 0
   
    
    n = []
    q = []
    
    
    w = 100
#     winsize = 0.1
    winsizes = np.arange(0.1,4.0,0.05)
    for winsize in winsizes:
       
        count1 = 0
        

        for m in range(1000):
            A = NonHomogenousPoissonProcess(T)
            B = NonHomo(T)
            C = HomogenousPoissonProcess(T, 20)
       
            dist3 = DTWDistance(CountEventNum(A,winsize),CountEventNum(B,winsize),w)
            dist4 = DTWDistance(CountEventNum(A,winsize),CountEventNum(C,winsize),w)
#             print(dist3)
        
            if dist3 > dist4:
                count1 += 1
            
#         q.append(count1/1000)
        q.append(dist3)
#         print (q)
        

#     print (dist3)

    plt.plot(winsizes,q)
    plt.xlabel('winsize')
    plt.ylabel('error rate')

    plt.show()

if __name__ == '__main__':
    run()
