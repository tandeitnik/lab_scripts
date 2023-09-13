# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:00:06 2023

@author: tandeitnik
"""

from __future__ import print_function
import libtiepie
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.signal as signal

#####################################################################
#####################################################################

#TiePie configurations

freq = int(5e5) #sampling frequency [Hz]
acqTime = 0.25 # total acquisiton time [s]
coupling= "ACV" #coupling type, can be ACV or DCV.
voltageRange = 2 #oscilloscope range
channel_1 = 1 # 0 = off, 1 = on
channel_2 = 1 # 0 = off, 1 = on

#PSD configurations

windows = 30 #number of windows used for the welch method

#####################################################################
#####################################################################

#connecting to tiepie
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
dt = 1/freq
recordLength = int(acqTime/dt)
scp.record_length = recordLength

# For all channels:
for ch in scp.channels:
    # Enable channel to measure it:
    ch.enabled = True

    # Set range:
    ch.range = voltageRange

    # Set coupling:
    if coupling == "ACV":
        ch.coupling = libtiepie.CK_ACV
    else:
        ch.coupling = libtiepie.CK_DCV

#getting first data and setting up the plot

# Start measurement:
scp.start()

# Wait for measurement to complete:
while not (scp.is_data_ready or scp.is_data_overflow):
    time.sleep(0.01)  # 10 ms delay, to save CPU time

# Get data:
data = scp.get_data()

# Stop stream:
scp.stop()

if channel_1 == 1 and channel_2 == 0:

    f, power = signal.welch(data[0], freq, window = 'hamming', nperseg = int(len(data[0])/windows))
    
    plt.ion()
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1, hspace=0.2, wspace=0)
    (ax1) = gs.subplots()
    line1, = ax1.plot(f/1000,power)
    #ax1.set_xlim([10, f/2])
    #ax1.set_ylim([min(power[1:]), max(power)])
    ax1.set_yscale('log')
    ax1.set(ylabel=r'$V^2$/Hz')
    ax1.set(xlabel=r'freq [kHz]')
    ax1.grid(alpha = 0.4)
    ax1.set(title = "PSD channel 1")
    
    stop = False
    
    while not stop:
        
        # Start measurement:
        scp.start()
    
        # Wait for measurement to complete:
        while not (scp.is_data_ready or scp.is_data_overflow):
            time.sleep(0.01)  # 10 ms delay, to save CPU time
    
        # Get data:
        data = scp.get_data()
    
        # Stop stream:
        scp.stop()
        
        f, power = signal.welch(data[0], freq, window = 'hamming', nperseg = int(len(data[0])/windows))
        
        line1.set_ydata(power)
        fig.canvas.draw()
        fig.canvas.flush_events()
        #time.sleep(0.1)
        
elif channel_1 == 0 and channel_2 == 1:
    
    f, power = signal.welch(data[1], freq, window = 'hamming', nperseg = int(len(data[1])/windows))
    
    plt.ion()
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1, hspace=0.2, wspace=0)
    (ax1) = gs.subplots()
    line1, = ax1.plot(f/1000,power)
    #ax1.set_xlim([10, f/2])
    #ax1.set_ylim([min(power[1:]), max(power)])
    ax1.set_yscale('log')
    ax1.set(ylabel=r'$V^2$/Hz')
    ax1.set(xlabel=r'freq [kHz]')
    ax1.grid(alpha = 0.4)
    ax1.set(title = "PSD channel 2")
    
    stop = False
    
    while not stop:
        
        # Start measurement:
        scp.start()
    
        # Wait for measurement to complete:
        while not (scp.is_data_ready or scp.is_data_overflow):
            time.sleep(0.01)  # 10 ms delay, to save CPU time
    
        # Get data:
        data = scp.get_data()
    
        # Stop stream:
        scp.stop()
        
        f, power = signal.welch(data[1], freq, window = 'hamming', nperseg = int(len(data[1])/windows))
        
        line1.set_ydata(power)
        fig.canvas.draw()
        fig.canvas.flush_events()
        #time.sleep(0.1)

elif channel_1 == 1 and channel_2 == 1:
    
    f, power_1 = signal.welch(data[0], freq, window = 'hamming', nperseg = int(len(data[0])/windows))
    f, power_2 = signal.welch(data[1], freq, window = 'hamming', nperseg = int(len(data[1])/windows))
    
    plt.ion()
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2, hspace=0.2, wspace=0.2)
    (ax1, ax2) = gs.subplots()
    
    #channel 1
    line1, = ax1.plot(f/1000,power_1)
    #ax1.set_xlim([10, f/2])
    #ax1.set_ylim([min(power[1:]), max(power)])
    ax1.set_yscale('log')
    ax1.set(ylabel=r'$V^2$/Hz')
    ax1.set(xlabel=r'freq [kHz]')
    ax1.grid(alpha = 0.4)
    ax1.set(title = "PSD channel 1")
    
    #channel 2
    line2, = ax2.plot(f/1000,power_2)
    #ax1.set_xlim([10, f/2])
    #ax1.set_ylim([min(power[1:]), max(power)])
    ax2.set_yscale('log')
    ax2.set(ylabel=r'$V^2$/Hz')
    ax2.set(xlabel=r'freq [kHz]')
    ax2.grid(alpha = 0.4)
    ax2.set(title = "PSD channel 2")
    
    stop = False
    
    while not stop:
        
        # Start measurement:
        scp.start()
    
        # Wait for measurement to complete:
        while not (scp.is_data_ready or scp.is_data_overflow):
            time.sleep(0.01)  # 10 ms delay, to save CPU time
    
        # Get data:
        data = scp.get_data()
    
        # Stop stream:
        scp.stop()
        
        f, power_1 = signal.welch(data[0], freq, window = 'hamming', nperseg = int(len(data[0])/windows))
        f, power_2 = signal.welch(data[1], freq, window = 'hamming', nperseg = int(len(data[1])/windows))
        
        line1.set_ydata(power_1)
        line2.set_ydata(power_2)
        fig.canvas.draw()
        fig.canvas.flush_events()
        #time.sleep(0.1)
    
    
