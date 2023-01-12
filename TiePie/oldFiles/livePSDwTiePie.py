# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import os
import libtiepie
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from printinfo import *
import keyboard


freq = int(2e6) #sampling frequency [Hz]
acqTime = 0.05 # total acquisiton time [s]
outputFolder = r"C:\Users\Labq\Documents\tandeitnik\savedFiles" #output folder where data will be saved
saveData = 0 #if 0 no data is saved, if 1 the data is saved in the outputFolder
N = 1 # número de traços
windows = 15 #number of windows used in the PSD evaluation
channelPSD = 1 #which channel to plot PSD, 1 for channel 1 and 2 for channel 2.


dt = 1/freq
recordLength = int(acqTime/dt)
#delay = 0.5


"""tie pie stuff [begin]"""
libtiepie.network.auto_detect_enabled = True

# Search for devices:
libtiepie.device_list.update()

# Try to open an oscilloscope with stream measurement support:
scp = None
for item in libtiepie.device_list:
    if item.can_open(libtiepie.DEVICETYPE_OSCILLOSCOPE):
        scp = item.open_oscilloscope()
        if scp.measure_modes & libtiepie.MM_STREAM:
            break
        else:
            scp = None
            
assert scp != None, "OSCILLOSCOPE NOT FOUND"

scp.measure_mode = libtiepie.MM_STREAM

# Set sample frequency:
scp.sample_frequency = freq

# Set record length:
scp.record_length = recordLength

# For all channels:
for ch in scp.channels:
    # Enable channel to measure it:
    ch.enabled = True

    # Set range:
    ch.range = 10

    # Set coupling:
    ch.coupling = libtiepie.CK_DCV
""" """

""" preparing the plot canvas """
#fig = plt.figure()
#plt.rcParams.update({'font.size': 14})
#plt.rcParams["axes.linewidth"] = 1
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlabel(r'freq [Hz]')
plt.ylabel(r'$V^2$/Hz')

#plt.show()

""" """

count = 0 #will count how many loops happened

""" 
Attention: to stop the following code, press Ctrl+C
"""

condition = True

while condition:
    
    if keyboard.is_pressed("q"):
        plt.close()
        condition = False
        
    #getting the data
    dataList = []
    
    for n in range(N):
    
        # Start measurement:
        scp.start()
    
        # Wait for measurement to complete:
        while not (scp.is_data_ready or scp.is_data_overflow):
            time.sleep(0.01)  # 10 ms delay, to save CPU time
    
        # Get data:
        data = scp.get_data()
        size = len(data[0])
        #put data into an array (maybe I can skip this!)
        dataArray = np.zeros( [size,3] )
        dataArray[:,0] = np.linspace(0,acqTime,size) #Time
        dataArray[:,1] = data[0] #Channel 1
        dataArray[:,2] = data[1] #Channel 2
    
        dataList.append(dataArray)
    
        # Stop stream:
        scp.stop()
    
    #evaluating the PSD
    freqPSD, power = signal.welch(dataList[0][:,channelPSD], freq, window = 'hamming', nperseg = int(len(dataList[0][:,channelPSD])/windows))
    for i in range(1,N):
        
        freqPSD, powerTemp = signal.welch(dataList[i][:,channelPSD], freq, window = 'hamming', nperseg = int(len(dataList[i][:,channelPSD])/windows))
        power += powerTemp
        
    power = power/N
    
    #Plotting the PSD
    
    if count == 0:
        line1, = ax.plot(freqPSD ,power)
        ax.set_yscale('log')
        ax.set_xscale('log')
        #ax = plt.gca()
    else:
        line1.set_xdata(freqPSD)
        line1.set_ydata(power)
    
    	# re-drawing the figure
        fig.canvas.draw()
	
    	# to flush the GUI events
        fig.canvas.flush_events()
        time.sleep(0.001)
    
    
#        ax.set_yscale('log')
#        ax.set_xscale('log')
#        
#        ax.set(ylabel=r'$V^2$/Hz')
#        ax.set(xlabel=r'freq [Hz]')
#        ax.scatter(freqPSD ,power, s = 3)
#        plt.draw()
#        plt.pause(0.001)
    
    #Saving the data if it set to be saved
    if saveData == 1:
    
        #determining the name of the files
        fileNames = []
        
        for i in range(count*N,count*N+N):
            
            if i <= 9:
                
                fileNames.append('0'+str(i))
                
            else:
                
                fileNames.append(str(i))
        
        #saving the files
        for n in range(N):
        
            outputFile = os.path.join(outputFolder,fileNames[n])
            np.save(outputFile,dataList[n])
            
    count += 1

# Close oscilloscope:
del scp
