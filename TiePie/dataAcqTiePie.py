# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import os
import libtiepie
import numpy as np
from printinfo import *
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import winsound

#####################################################################
#####################################################################

reps = 1 #number of repetitions the data is collected - it is different from number of traces
delay = 10 #time in seconds the script waits beetween each repetition

freq = int(5e5) #sampling frequency [Hz]
acqTime = 0.25 # total acquisiton time [s]
N = 20 # number of traces
rootFolder = r"C:\Users\Labq\Desktop\Daniel R.T\Nova pasta\data\backward-noparticle" #output folder where folders containint the data will be saved
coupling= "ACV" #coupling type, can be ACV or DCV.
voltageRange = 1e-3 #oscilloscope range
autoRange = 1 #if it equals to 1, voltageRange will be ignored and an automatic range will be determined
gainAutoRange = 3 #multiplicative factor that determines the autoRange

experimentDescription = "Partícula pinçada a 10mbar. Experimento para medir o SNR do foward."

#####################################################################
#####################################################################

now = datetime.now()

#writing experience info txt
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
        
def generateOrderedNumbers(maxValue):
    
    if maxValue != 0:
        decimalPlaces = int(np.log10(maxValue))
        numberList = ["0"*(decimalPlaces+1)]
    else:
        numberList = ["0"]
    
    if maxValue > 1:
        
        stopTest = 0
        for dP in range(decimalPlaces+1):
            
            for numeral in range(10**(dP+1)-10**dP):
                
                number = (decimalPlaces-dP)*"0"+str(numeral+10**dP)
                numberList.append(number)
                
                if int(number) == (maxValue-1):
                    stopTest = 1
                    break
                
            if stopTest == 1:
                break
        
    return numberList

repNumberList = generateOrderedNumbers(reps)
tracesNumberList = generateOrderedNumbers(N)

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
        

#creating folder
folders = next(os.walk(rootFolder))[1]
folderNumberList = generateOrderedNumbers(len(folders)+1)

for i in range(len(folders)+1):
    
    if ("data_"+folderNumberList[i]) in folders:
        pass
    else:
        outputFolder = os.path.join(rootFolder,"data_"+folderNumberList[i])
        os.mkdir(outputFolder)
        break
        

for rep in range(reps):
    
    outputFolderRep = os.path.join(outputFolder,"rep"+repNumberList[rep])
    os.mkdir(outputFolderRep)
    dt = 1/freq

    dataList = []
    
    if autoRange == 1:
        
        scp.start()
    
        # Wait for measurement to complete:
        while not (scp.is_data_ready or scp.is_data_overflow):
            time.sleep(0.01)  # 10 ms delay, to save CPU time
    
        # Get data:
        data = scp.get_data()
        size = len(data[0])
    
        rangeStdCH1 = np.std(data[0])
        rangeStdCH2 = np.std(data[0])
        # Stop stream:
        scp.stop()
        
        autoRangeValue = max(rangeStdCH1,rangeStdCH2)*gainAutoRange
        ch.range = autoRangeValue
        
    
    print("acquiring data")
    
    for n in tqdm(range(N)):    
    
        # Start measurement:
        scp.start()
    
        # Wait for measurement to complete:
        while not (scp.is_data_ready or scp.is_data_overflow):
            time.sleep(0.01)  # 10 ms delay, to save CPU time
    
        # Get data:
        data = scp.get_data()
        
        # Stop stream:
        scp.stop()
        
        # Put data into a data frame
        size = len(data[0])
        df = pd.DataFrame({'t':np.linspace(0,acqTime,size), 'ch1':data[0], 'ch2':data[1]})
        
        outputFile = os.path.join(outputFolderRep,tracesNumberList[n]+'.pkl')
        df.to_pickle(outputFile)
        
        #delete original df to save space
        del df
        
        

    if rep != reps-1:
        print("zzzzzzzzz")
        time.sleep(delay)

# Close oscilloscope:
del scp

winsound.Beep (440, 1000)
