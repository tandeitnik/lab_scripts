#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:06:50 2022

@author: tandeitnik
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import butter, filtfilt

f = 2e6
windows = 5
lowcut = 38_000
highcut = 93_000
order = 4

pathFolders = "/Users/tandeitnik/Desktop/temp/newLensNewAlig"  #pasta raíz aonde contém os dados
folders = next(os.walk(pathFolders))[1]

selectedFolders = ['l228_20db']

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


powerList = []
powerListFiltered = []

for folder in folders:
    
    if folder in selectedFolders:
    
        path = pathFolders + '/' + folder   #pasta raíz aonde contém os dados
        files = next(os.walk(path))[2]
        
        data = []
        dataFiltered = []
        
        for file in files:
            
            if file == '.DS_Store':
                
                pass
            
            else:
    
                data.append( np.genfromtxt(path+'/'+file, delimiter=',', skip_header=0)[:,4])
                dataFiltered.append(butter_bandpass_filter(data[-1], lowcut, highcut, f, order))
                
        
        freq, power = signal.welch(data[0], f, window = 'hamming', nperseg = int(len(data[0])/windows))
        errors = 0
    
        for i in range(1,len(data)):
            
            
            freq, powerTemp = signal.welch(data[i], f, window = 'hamming', nperseg = int(len(data[i])/windows))
            if len(powerTemp) != len(power):
                errors += 1
            else:
                power += powerTemp
            
        power = power/(len(data)-errors)
        
        powerList.append(power)
        
        
        
        freq, power = signal.welch(dataFiltered[0], f, window = 'hamming', nperseg = int(len(dataFiltered[0])/windows))
        errors = 0
    
        for i in range(1,len(data)):
            
            freq, powerTemp = signal.welch(dataFiltered[i], f, window = 'hamming', nperseg = int(len(dataFiltered[i])/windows))
            if len(powerTemp) != len(power):
                errors += 1
            else:
                power += powerTemp
            
        power = power/(len(data)-errors)
        
        powerListFiltered.append(power)
    

selectedFoldersPlot = selectedFolders

fig = plt.figure()
plt.rcParams.update({'font.size': 14})
plt.rcParams["axes.linewidth"] = 1

ax = plt.gca()

for i, folder in enumerate(selectedFolders):
    
    if folder in selectedFoldersPlot:
    
        ax.scatter(freq,powerList[i],label = 'Without BP Filter' , s = 3)
        ax.scatter(freq ,powerListFiltered[i],label = 'With BP Filter' , s = 3)
        
        
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
ax.set(ylabel=r'$V^2$/Hz')
ax.set(xlabel=r'freq [Hz]')
ax.grid(alpha = 0.4)
