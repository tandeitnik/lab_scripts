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

windows = 10 #number of windows used for the welch method
channel = 1 #which channel to evaluate PSD
welchMethod = 0 #if welchMethod == 1, then Welch method is used (which is quicker but it is an estimation). Otherwise, periodogram is used.
pathFolders = r"C:\Users\tandeitnik\Downloads\dadosBreno-20230104T194622Z-001\dadosBreno"  #root folder where the reps folders are saved


folders = next(os.walk(pathFolders))[1]
selectedFolders = folders #you want to ignore some folders, you can pass here a list of the desirable folders


powerList = []

if 'f' in locals():
    del f

for folder in folders:
    
    if folder in selectedFolders:
        
        path = os.path.join(pathFolders,folder)
        files = next(os.walk(path))[2]
        
        data = []
        
        for file in files:
            
            if file == '.DS_Store':
                
                pass
            
            else:
                
                data.append(np.load(os.path.join(pathFolders,folder,file))[:,channel])
                
                if ('f' in locals()) == False:
                
                    f = 1/(np.load(os.path.join(pathFolders,folder,file))[1,0]-np.load(os.path.join(pathFolders,folder,file))[0,0])
        
        if welchMethod == 1:
        
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
            
        else:
            
            freq, power = signal.periodogram(data[0], f, scaling='density')
            
            errors = 0
    
            for i in range(1,len(data)):
                
                freq, powerTemp = signal.periodogram(data[i], f, scaling='density')
                if len(powerTemp) != len(power):
                    errors += 1
                else:
                    power += powerTemp
                
            power = power/(len(data)-errors)
            
            powerList.append(power)
    
selectedFoldersPlot = selectedFolders

fig = plt.figure()
plt.rcParams.update({'font.size': 14})
plt.rcParams["axes.linewidth"] = 1

ax = plt.gca()

for i, folder in enumerate(selectedFolders):
        
    
    if folder in selectedFoldersPlot:
    
        #ax.scatter(freq ,powerList[i],label = folder , s = 10)
        ax.set_xlim([10, f/2])
        ax.set_ylim([min(powerList[i][1:]), max(powerList[i])])
        ax.plot(freq ,powerList[i],label = folder)
        
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
ax.set(ylabel=r'$V^2$/Hz')
ax.set(xlabel=r'freq [Hz]')
ax.grid(alpha = 0.4)
