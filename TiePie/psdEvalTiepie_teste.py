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
import pandas as pd

#####################################################################
#####################################################################

windows = 10 #number of windows used for the welch method
channel = 'ch1' #which channel to evaluate PSD, can be 'ch1' or 'ch2'
welchMethod = 1 #if welchMethod == 1, then Welch method is used (which is quicker but it is an estimation). Otherwise, periodogram is used.
pathFolders = r"C:\Users\tandeitnik\Downloads\dadosBreno-20230104T194622Z-001\dadosBreno"  #root folder where the reps folders are saved

folders = next(os.walk(pathFolders))[1] #don't modify this line!!
selectedFolders = folders #if you want to ignore some folders, you can pass here a list of the desirable folders. Else, just leave selectedFolders = folders

#####################################################################
#####################################################################

#list that will contain the PSD for each folder in the root folder
powerList = []

#this loop scans the folders in the root folder
for folder in folders:
    
    #the loop only makes something if the folder is in the selectedFolders list
    if folder in selectedFolders:
        
        #making the path name to desired folder
        path = os.path.join(pathFolders,folder)
        #getting the file names
        files = next(os.walk(path))[2]
        
        #this list stores the data frames found in the folder
        data = []
        
        #loop to scan the files in the folder
        for file in files:
            
            #test if the file is in right format
            if file.endswith('.pkl'):
                
                #making the filePath
                filePath = os.path.join(pathFolders,folder,file)
                #loading file and storing it into data
                data.append(pd.read_pickle(filePath))
        
        #determining the sampling frequency of the data
        dt = data[0].t[1]-data[0].t[0]
        f = 1/dt
        
        #evaluating the PSD
        if welchMethod == 1: #if welch method is ON
            
            #evaluates the PSD for the first trace
            freq, power = signal.welch(data[0][channel], f, window = 'hamming', nperseg = int(len(data[0][channel])/windows))
        
            #evaluates the PSD for the subsequent traces and sum
            for i in range(1,len(data)):
                freq, powerTemp = signal.welch(data[i][channel], f, window = 'hamming', nperseg = int(len(data[i][channel])/windows))
                power += powerTemp
                
            
        else: #if welch method is OFF
            
            #evaluates the PSD for the first trace
            freq, power = signal.periodogram(data[0][channel], f, scaling='density')
            
            #evaluates the PSD for the subsequent traces and sum
            for i in range(1,len(data)):
                
                freq, powerTemp = signal.periodogram(data[i], f, scaling='density')
                power += powerTemp
                
        #calculating the avarage
        power = power/len(data)
        #appending the mean PSD
        powerList.append(power)
        
##################
#PLOTTING SECTION#
##################
    
#if you want, you can select a subset of folders to plot
selectedFoldersPlot = selectedFolders

fig = plt.figure()
plt.rcParams.update({'font.size': 14})
plt.rcParams["axes.linewidth"] = 1

ax = plt.gca()

for i, folder in enumerate(selectedFolders):
        
    
    if folder in selectedFoldersPlot:
    
        #ax.scatter(freq ,powerList[i],label = folder , s = 10)
        ax.plot(freq ,powerList[i],label = folder)
        ax.set_xlim([10, f/2])
        ax.set_ylim([min(powerList[i][1:]), max(powerList[i])])
        
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
ax.set(ylabel=r'$V^2$/Hz')
ax.set(xlabel=r'freq [Hz]')
ax.grid(alpha = 0.4)
