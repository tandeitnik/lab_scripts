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

f = 2e6
windows = 5
pathFolders = "/Users/tandeitnik/Desktop/temp/newLensNewAlig/withVoltage"  #root folder where the reps folders are saved


folders = next(os.walk(pathFolders))[1]
selectedFolders = folders #you want to ignore some folders, you can pass here a list of the desirable folders


powerList = []


for folder in folders:
    
    if folder in selectedFolders:
        
        path = os.path.join(pathFolders,folder)   #pasta raíz aonde contém os dados
        files = next(os.walk(path))[2]
        
        data = []
        
        for file in files:
            
            if file == '.DS_Store':
                
                pass
            
            else:

                data.append( np.genfromtxt(os.path.join(pathFolders,folder,file), delimiter=',', skip_header=0)[:,4])
                
        
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
        ax.plot(freq ,powerList[0],label = folder)
        
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
ax.set(ylabel=r'$V^2$/Hz')
ax.set(xlabel=r'freq [Hz]')
ax.grid(alpha = 0.4)
        
