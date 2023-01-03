# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import os
import libtiepie
import numpy as np
from printinfo import *
from datetime import datetime
from tqdm import tqdm
import beepy
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import stats

#####################################################################
#####################################################################

#EXPERIMENT PARAMETERS
reps = 20 #number of voltages the data is collected - it is different from number of traces
drivingFreq = 83e3 #driving harmonic frequency
freqRange = 100

#OSCILLOSCOPE PARAMETERS
freq = int(200e3) #sampling frequency [Hz]
acqTime = 1 # total acquisiton time [s]
N = 20 # number of traces
rootFolder = r"C:\Users\tandeitnik\Downloads\electricForceCalibration-20230103T185658Z-001\electricForceCalibration\teste0" #output folder where folders containint the data will be saved
coupling= "ACV" #coupling type, can be ACV or DCV.
voltageRange = 1e-3 #oscilloscope range
autoRange = 1 #if it equals to 1, voltageRange will be ignored and an automatic range will be determined
gainAutoRange = 1.5 #multiplicative factor that determines the autoRange
experimentDescription = "Teste para ver se funciona." #write relevant information about the experiment

#ELECTRIC FORCE ESTIMATION PARAMETERS
windows = 1 #number of windows used for the welch method
channel = 1 #which channel to evaluate PSD
welchMethod = 1 #if welchMethod == 1, then Welch method is used (which is quicker but it is an estimation). Otherwise, periodogram is used.


#####################################################################
#####################################################################


#writing experience info txt

now = datetime.now()

lines = ['Experiment info',
         '',
         'Date and time: '+now.strftime("%d/%m/%Y %H:%M:%S"),
         'Device: TiePie',
         'Samp. freq.: '+str(freq),
         'acq. time.: '+str(acqTime),
         'Num. traces: '+str(N),
         'Coupling: '+coupling,
         'Description: '+experimentDescription
         ]

with open(os.path.join(rootFolder,'experimentInfo.txt'), 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')


voltageValues = np.zeros(reps)

def generateOrderedNumbers(maxValue):

    decimalPlaces = int(np.log10(maxValue))
    stopTest = 0
    numberList = ["0"*(decimalPlaces+1)]
    
    for dP in range(decimalPlaces+1):
        
        for numeral in range(10**(dP+1)-10**dP):
            
            number = (decimalPlaces-dP)*"0"+str(numeral+10**dP)
            numberList.append(number)
            
            if number == str(maxValue-1):
                stopTest = 1
                break
            
        if stopTest == 1:
            break
        
    return numberList

numberList = generateOrderedNumbers(reps)

for rep in range(reps):
    
    voltageValues[rep] = float(input("Type applyed voltage amplitude (peak to peak): "))
    input("Press ENTER to get next set of measurements...")
    
    outputFolder = os.path.join(rootFolder,"rep"+numberList[rep])
    os.mkdir(outputFolder)
    dt = 1/freq
    recordLength = int(acqTime/dt)

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
        ch.range = voltageRange
    
        # Set coupling:
        if coupling == "ACV":
            ch.coupling = libtiepie.CK_ACV
        else:
            ch.coupling = libtiepie.CK_DCV
    
    dataList = []
    
    if autoRange == 1:
        
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
    
        rangeStdCH1 = np.std(data[0])
        rangeStdCH2 = np.std(data[0])
        # Stop stream:
        scp.stop()
        
        autoRangeValue = max(rangeStdCH1,rangeStdCH1)*gainAutoRange
        ch.range = autoRangeValue
        print("Range set to "+str(autoRangeValue)+"V")
        
    
    print("acquiring data")
    
    for n in tqdm(range(N)):    
    
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
    
    print("saving data")
    
    for n in tqdm(range(N)):
        
        if n <= 9:
            
            outputFile = os.path.join(outputFolder,'0'+str(n))
            np.save(outputFile,dataList[n])
            
        else:
            
            outputFile = os.path.join(outputFolder,str(n))
            np.save(outputFile,dataList[n])
            
    del dataList
    beepy.beep(sound=1)
    

# Close oscilloscope:
del scp

#Discovering the force for each measurement

folders = next(os.walk(rootFolder))[1]

powerList = []

if 'f' in locals():
    del f

for folder in folders:
    
    print(folder)
    
    path = os.path.join(rootFolder,folder)
    files = next(os.walk(path))[2]
    
    data = []
    
    for file in files:
        
        if file == '.DS_Store':
            
            pass
        
        else:
            
            data.append(np.load(os.path.join(rootFolder,folder,file))[:,channel])
            
            if ('f' in locals()) == False:
            
                f = 1/(np.load(os.path.join(rootFolder,folder,file))[1,0]-np.load(os.path.join(rootFolder,folder,file))[0,0])
    
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
        
deltaFreq = freq[1] -freq[0]
indCentral = int(drivingFreq/deltaFreq)
idxLeft = int(indCentral-(freqRange/deltaFreq))
idxRight = int(indCentral+(freqRange/deltaFreq))
top = np.zeros(reps)
bottom = np.zeros(reps)
height = np.zeros(reps)

for i in range(reps):
    
    top[i] = max(powerList[i][idxLeft:idxRight])
    bottom[i] = (powerList[i][idxLeft]+powerList[i][idxRight])/2
    height[i] = top[i] - bottom[i]
    
    plt.loglog(freq,powerList[i])
    plt.loglog(freq,np.ones(len(freq))*top[i])
    plt.loglog(freq,np.ones(len(freq))*bottom[i])
    
#Linear regression
slope, intercept, r, p, se = stats.linregress(voltageValues, height)

plt.plot(slope*voltageValues+intercept)
plt.scatter(voltageValues, height)

