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
from scipy.fft import dct
import os

#This block records stuff from mic to save into file, not relevant for CW
#seconds = 5
#r = sd.rec(seconds * fs, samplerate=fs, channels=1)
#sd.wait()
#sf.write('speech1.wav', r, fs)

#r_in, fs_in = sf.read(r"soundData/Sivaprasath/sivaprasath_0.wav", dtype='float32')
#sd.play(r_in, fs_in)
#x = r_in.squeeze()

#============== Vars ===============
baseDir = r"soundData"
outputDir = r"mfccs"
os.makedirs(outputDir, exist_ok=True)
#numSamples = len(x)
fs = 16000
M = 8 
N = 320
H = N // 2
W = np.hamming(N)
num_ceps = min(13, M) 
mfcc_frames = []

#============= Filterbank Function =============

def melFilterBank(magspec, fs, M):
    fMin = 0 #p much will always be zero
    fMax = fs / 2 #8000 in the case of 16000Hz
    eps = 1e-10 
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
            
    melEnergies = np.matmul(H, magspec) #derive mel energies 
    logMel = np.log(melEnergies + eps) #take the log of these mel energies and return
    
    
    return logMel

#========= Extract Frames ============

for root, _, files in os.walk(baseDir):
    for file in files:
        if file.lower().endswith(".wav"):
            filePath = os.path.join(root, file)
            try:
                x, fs_in = sf.read(filePath, dtype="float32")
                x = x.squeeze()
                
                frames = []
                for i in range(0, len(x), H): #iteration step is set to H which is our frame size // 2, meaning we have overlap
                    frame = x[i:i+N] 
                    if len(frame) == N:
                        frames.append(frame)
                frames = np.array(frames)
                    
#========= Extract MFCC ============
                mfccFrames = []
                for frame in frames:
                    magSpec = np.abs(np.fft.rfft(frame)) #find magspec for each frame
                    logMel = melFilterBank(magSpec, fs_in, M) #then put it through mel filterbank and return log of mel energies
                    ceps = dct(logMel, type=2, norm='ortho')[:num_ceps] #apply DCT
                    mfccFrames.append(ceps) #then add to final MFCC array
                    
                mfcc = np.vstack(mfccFrames)
                    
#========= Save output based on name ============
                speakerName = os.path.basename(root)
                outName = f"{os.path.splitext(file)[0]}.npy"
                outPath = os.path.join(outputDir, outName)
                np.save(outPath, mfcc)
                print(f"{outName} has been saved")

            except Exception as e:
                print(f"Error processing {filePath}: {e}")

print("Done")
                    
                              
            

        
"""
OLD STUFF HERE JUST FOR ARCHIVING PURPOSES
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
                
    logMel = melFilterBank(magSpec, fs, M)
    
    ceps = dct(logMel, type=2, norm='ortho')[:num_ceps]
    mfcc_frames.append(ceps)
    
    
mfcc = np.vstack(mfcc_frames)
np.save('mfcc.npy', mfcc)
print('MFCC shape:', mfcc.shape) 

#mfccToPlot = mfcc[:, 1:]

#eps = 1e-8
#m = mfccToPlot.mean(axis=0, keepdims=True)
#s = mfccToPlot.std(axis=0, keepdims=True)
#mfccNorm = (mfccToPlot - m) / (s + eps)

#plt.figure(figsize=(10, 4))
#plt.imshow(mfccNorm.T, aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
#plt.title("MFCC Feature Matrix")
#plt.xlabel("Frame index (time →)")
#plt.ylabel("Cepstral coefficient")
#plt.colorbar(label="Normalized amplitude")
#plt.tight_layout()
#plt.show()
"""