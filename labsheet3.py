# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 13:47:19 2025

@author: Admin
"""

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import plotly.io as pio
import plotly.express as px
import pandas as pd
import soundfile as sf
import math


fs = 16000
seconds = 5


#r = sd.rec(seconds * fs, samplerate=fs, channels=1)
#sd.wait()
#sf.write('speech.wav', r, fs)
#^^ UNCOMMENT THESE LINES IF YOU WANT TO RE-RECORD "SPEECH.WAV"
#COMMENT THEM OUT OTHERWISE IF YOU WANT TO KEEP OPERATING ON THE SAME WAV FILE

r_in, fs_in = sf.read('speech.wav', dtype='float32')

sd.play(r_in, fs_in)

x = r_in.squeeze()

numSamples = len(x)
N = 320
H = N // 2
numFrames = numSamples // N 
W = np.hamming(N)
f = np.fft.rfftfreq(N, d=1/fs_in)

"""
#r = np.zeros(256)
#r[0:32] = 1
#r[32:64] = 2
#r[64:96] = 3
#r[96:128] = 4
#r[128:160] = 5
#r[160:192] = 6
#r[192:224] = 7
#r[224:256] = 8

#fbank = np.zeros(8)
"""
"""
#def linearRectangularFilterbank(magspec, M): #THIS IS OLD, JUST HERE FOR ARCHIVING
    #magspec = np.asarray(magspec)
    #K = len(magspec)
    #edges = np.linspace(0, K, M + 1, dtype=int)
    #out = np.zeros(M, dtype=magspec.dtype)
    #for m in range(M):
        #out[m] = magspec[edges[m]:edges[m+1]].sum()
    #return out
"""    
def linearRectangularFilterbank(magspec, M): #CURRENT MATRIX MULTIPLICATION VERSION
    magspec = np.asarray(magspec)
    K = magspec.size
    edges = np.linspace(0, K, M + 1, dtype=int)

    out = np.zeros(M, dtype=magspec.dtype)

    r = np.zeros(K, dtype=float)
    for m in range(M):
        r[:] = 0.0
        r[edges[m]:edges[m+1]] = 1.0   
        out[m] = np.matmul(r, magspec)           
    return out

def melFilterBank(magspec, fs, M):
    fMin = 0 #p much will always be zero
    fMax = fs / 2 #8000 in the case of 16000Hz
    melMin = 1125 * np.log(1 + fMin / 700) 
    melMax = 1125 * np.log(1 + fMax / 700) #calculating mel frequencies for min and max (melmin should be zero but keeping that for flexibility)
    melPoints = np.linspace(melMin, melMax, M + 2) #equally spaced mel points by approx 315
    hzPoints = 700.0 * (np.exp(melPoints / 1125.0) - 1.0) #convert back into unevenly spaced Hz points
    K = magspec.size 
    originalN = (K - 1) * 2 #derive original FFT window size before rfft 
    freqSpacing = fs / originalN #Hz per bin
    
    binPoints = np.floor(hzPoints / freqSpacing).astype(int) #derive fft bin indices
    
    H = np.zeros((M, K), dtype=float)
    
    for m in range(1, M + 1):   # getting centres
        left   = binPoints[m - 1]
        centre = binPoints[m]
        right  = binPoints[m + 1]
        
        #rising slope implementation
        if centre > left:
            k = np.arange(left, centre)
            H[m-1, k] = (k - left) / (centre - left)
        
        #define peak of triangle
        H[m-1, centre] = 1.0
        
        #falling slope implementation
        if right > centre:
            k = np.arange(centre + 1, right + 1)
            H[m-1, k] = (right - k) / (right - centre)
            
    melEnergies = np.matmul(H, magspec)
    
    
    return melPoints, hzPoints, binPoints, melEnergies, H

frames = []

for i in range(0, len(x), H):
    frame = x[i:i+N]
    if len(frame) == N:
        frames.append(frame)
        
frames = np.array(frames)

for i in range(len(frames)):
    frames[i] = frames[i] * W

for i, frame in enumerate(frames):
    frameFFT = np.fft.rfft(frame)
    magSpec  = np.abs(frameFFT)
    phaseSpec= np.angle(frameFFT)
    
    M = 8                     
    fbank = linearRectangularFilterbank(magSpec, 8) #PASS INTO FUNCTION
    melPt, hzPt, binPt, melEnrgy, hTest = melFilterBank(magSpec, fs, M)
    
    """
    #K = magSpec.size           
    #bins_per_band = K // M
    
    #r = np.zeros(K)
    #r[2*bins_per_band : 3*bins_per_band] = 1.0
    
    #fbankMatmul = np.zeros(M)
    #fbankMatmul[2] = np.matmul(r, magSpec)
    """    
    """
    #fbank[0] = sum(magSpec[0:32])
    #fbank[1] = sum(magSpec[32:64])
    #fbank[2] = sum(magSpec[64:96])
    #fbank[3] = sum(magSpec[96:128])
    #fbank[4] = sum(magSpec[128:160])
    #fbank[5] = sum(magSpec[160:192])
    #fbank[6] = sum(magSpec[192:224])
    #fbank[7] = sum(magSpec[224:256])
    """
    #changes ---> function replaced the hardcoded filterbank boundaries
    #pass in the correct amount
    
    #UNCOMMENT THIS BLOCK IF YOU WANT A MAGSPEC PLOT
    """
    #fig, axs = plt.subplots(2)
    #axs[0].plot(f, magSpec)            # use Hz on x-axis
    #axs[0].set_xlabel("Frequency (Hz)")
    #axs[0].set_ylabel("Magnitude")
    #axs[1].plot(fbank)
    #axs[1].set_xlabel("Band index (0..7)")
    #axs[1].set_ylabel("Band sum")
    #plt.show()
    """

    
    
    
    
    



