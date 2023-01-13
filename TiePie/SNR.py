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
import scipy.signal as signal
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties import unumpy
import winsound

####################
#PARAMETERS SECTION#
####################


#oscilloscope setup
#####################

f = int(1e6) #sampling frequency [Hz]
acqTime = 0.1 # total acquisiton time [s]
N = 200 # number of traces
rootFolder = r"C:\Users\Labq\Desktop\Daniel R.T\Nova pasta\calibrationTest" #output folder where the calibration data will be saved
coupling= "ACV" #coupling type, can be ACV or DCV.
voltageRange = 1e-3 #oscilloscope range
autoRange = 1 #if it equals to 1, voltageRange will be ignored and an automatic range will be determined
gainAutoRange = 3 #multiplicative factor that determines the autoRange

#write a description of the experiment
experimentDescription = "SNR. Vacumm chamber at XXmbar."

saveRawData = 0 #if 1 the raw data is saved, else the raw data is deleted and only the mean PSD is saved

#PSD setup
#####################

windows = 10 #number of windows used for the welch method
channel = 'ch1' #which channel to evaluate PSD, can be 'ch1' or 'ch2'
welchMethod = 1 #if welchMethod == 1, then Welch method is used (which is quicker but it is an estimation). Otherwise, periodogram is used.


######################
#connecting to TiePie#
######################


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
scp.sample_frequency = f

# Set record length:
dt = 1/f
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

####################
#acquiring the data#
####################

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

tracesNumberList = generateOrderedNumbers(N)

folders = next(os.walk(rootFolder))[1]
folderNumberList = generateOrderedNumbers(len(folders)+1)

for i in range(len(folders)+1):
    
    if ("SNR_"+folderNumberList[i]) in folders:
        pass
    else:
        outputFolder = os.path.join(rootFolder,"SNR_"+folderNumberList[i])
        os.mkdir(outputFolder)
        break


print("acquiring data")

for snr in range(3):
    
    winsound.Beep (440, 1000)
    
    if snr == 0:
    
        print("\nFirst, get the data of the detector without any incident lasers...")
        input("\nWhen ready, press ENTER to collect data.")
        
    elif snr == 1:
        
        print("\nSecond, get the data with the laser focusing on the detector, but without a trapped particle...")
        input("\nWhen ready, press ENTER to collect data.")
        
    elif snr == 2:
        
        print("\nThird, get the data with a trapped particle at a low pressure...")
        input("\nWhen ready, press ENTER to collect data.")
    
    
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
        
        #Calculating PSD
        if welchMethod == 1: #if welch method is ON
        
            if n == 0: #first round
        
                freq, power = signal.welch(df[channel], f, window = 'hamming', nperseg = int(len(df[channel])/windows))
                powerArray = np.zeros([N,len(freq)])
                powerArray[0,:] = power
                
            else: #subsequent rounds
            
                freq, power = signal.welch(df[channel], f, window = 'hamming', nperseg = int(len(df[channel])/windows))
                powerArray[n,:] = power
        
            
        else: #if welch method is OFF
            
            if n == 0: #first round
            
                #evaluates the PSD for the first trace
                freq, power = signal.periodogram(df[channel], f, scaling='density')
                powerArray = np.zeros([N,len(freq)])
                powerArray[0,:] = power
            
            else: #subsequent rounds
    
                
                freq, power = signal.periodogram(df[channel], f, scaling='density')
                powerArray[n,:] = power
    
        #################
        #saving the data#
        #################
    
        if saveRawData == 1:
            
            outputFile = os.path.join(outputFolder,tracesNumberList[n])
            df.to_pickle(outputFile)
    
        #delete original df to save space
        del df
        
        # Calculate mean PSD and standard error
        meanPSD = unumpy.uarray( np.mean(powerArray, axis = 0) , np.std(powerArray,axis = 0) )
        
        if snr == 0:
        
            df = pd.DataFrame({'f [Hz]':freq, 'Floor PSD [V**2/Hz]':meanPSD})
            
        elif snr == 1:
            
            df['Laser PSD [V**2/Hz]'] = meanPSD
            
        elif snr == 2:
            
            df['Particle PSD [V**2/Hz]'] = meanPSD

        del powerArray, meanPSD

# Close oscilloscope:
del scp

#saving the data frame with the PSDs
outputFile = os.path.join(outputFolder,'PSD.pkl')
df.to_pickle(outputFile)

#################################################
#getting left and right frequency cuts from user#
#################################################
        
#printing result so that the user may decide where to trimm the PSD
dpi = 100
fig = plt.figure(figsize=(1.5*1080/dpi,1.5*720/dpi), dpi=dpi)
plt.rcParams.update({'font.size': 24})
plt.rcParams["axes.linewidth"] = 1

ax = plt.gca()

ax.scatter(df['f [Hz]'] ,unumpy.nominal_values(df['Floor PSD [V**2/Hz]']), s = 10, label = 'floor')
ax.scatter(df['f [Hz]'] ,unumpy.nominal_values(df['Laser PSD [V**2/Hz]']), s = 10, label = 'laser')
ax.scatter(df['f [Hz]'] ,unumpy.nominal_values(df['Particle PSD [V**2/Hz]']), s = 10, label = 'particle')
ax.set_ylim([min(unumpy.nominal_values(df['power [V**2/Hz]'][1:])), 2*max(unumpy.nominal_values(df['Laser PSD [V**2/Hz]']))])
ax.set_xlim([1000, f/2])
ax.legend()
ax.set_yscale('log')
ax.set_xscale('log')
ax.set(ylabel=r'$V^2$/Hz')
ax.set(xlabel=r'$\Omega/2\pi$ [Hz]')
ax.set_axisbelow(True)
ax.minorticks_on()
ax.grid(which='major', linestyle='-', linewidth='1')
ax.grid(which='minor', linestyle=':', linewidth='1')
plt.tight_layout()

outputFile = os.path.join(outputFolder,'fullPSD.png')
plt.savefig(outputFile)
plt.close()

winsound.Beep (440, 1000)
print("\nPlease, open the image file 'fullPSD.png' in "+outputFolder)
print("\nFrom the plot, you should select a minimum and maximum frequencies to cut the PSD.")

agreementTest = 0

while agreementTest == 0:
    
    leftCut = int(input("\nEnter a frequency value for the minimum frequency and press ENTER: "))
    rightCut = int(input("\nEnter a frequency value for the maximum frequency and press ENTER: "))
    print("\nYou entered "+str(leftCut)+"Hz for the minimum frequency and "+str(rightCut)+"Hz for the maximum frequency.")
    agree = input("Do you agree? [y/n]")
    
    while (agree != 'y') and  (agree != 'n'):
        agree = input("\ninvalid input, please enter y if you agree or n if you disagree and want to replace the values: ")

    if agree == 'y':
        agreementTest = 1
    
#trimming the PSD

deltaFreq = freq[1]-freq[0]
idxLeft = int(leftCut/deltaFreq)
idxRight = int(rightCut/deltaFreq)

trimmedPSD = df[idxLeft:idxRight].reset_index()
#saving the data frame with trimmed PSD
outputFile = os.path.join(outputFolder,'trimmedPSD.pkl')
trimmedPSD.to_pickle(outputFile)

################
#evaluating SNR#
################

#finding max value of the trapped particle signal
partSignal = unumpy.nominal_values(max(trimmedPSD['Particle PSD [V**2/Hz]']))

#evaluating the mean signal strength for floor and laser noise
floorNoise = np.mean(unumpy.nominal_values(trimmedPSD['Floor PSD [V**2/Hz]']))
laserNoise = np.mean(unumpy.nominal_values(trimmedPSD['Laser PSD [V**2/Hz]']))

#evaluating SNRs
SNR_floor = 10*np.log10(partSignal/floorNoise)
SNR_laser = 10*np.log10(partSignal/laserNoise)

########################################
#saving results and showing to the user#
########################################

dpi = 100
fig = plt.figure(figsize=(1.5*1080/dpi,1.5*720/dpi), dpi=dpi)
plt.rcParams.update({'font.size': 24})
plt.rcParams["axes.linewidth"] = 1

ax = plt.gca()

ax.scatter(trimmedPSD['f [Hz]'] ,unumpy.nominal_values(trimmedPSD['Floor PSD [V**2/Hz]']), s = 10, label = 'floor - SNR = '+str(int(SNR_floor))+'db')
ax.scatter(trimmedPSD['f [Hz]'] ,unumpy.nominal_values(trimmedPSD['Laser PSD [V**2/Hz]']), s = 10, label = 'laser - SNR = '+str(int(SNR_laser))+'db')
ax.scatter(trimmedPSD['f [Hz]'] ,unumpy.nominal_values(trimmedPSD['Particle PSD [V**2/Hz]']), s = 10, label = 'particle')
ax.set_ylim([min(unumpy.nominal_values(trimmedPSD['power [V**2/Hz]'][1:])), 2*max(unumpy.nominal_values(trimmedPSD['Laser PSD [V**2/Hz]']))])
ax.set_xlim([leftCut, rightCut])
ax.legend()
ax.set_yscale('log')
ax.set_xscale('log')
ax.set(ylabel=r'$V^2$/Hz')
ax.set(xlabel=r'$\Omega/2\pi$ [Hz]')
ax.set_axisbelow(True)
ax.minorticks_on()
ax.grid(which='major', linestyle='-', linewidth='1')
ax.grid(which='minor', linestyle=':', linewidth='1')
plt.tight_layout()

outputFile = os.path.join(outputFolder,'SNR_PSDs.png')
plt.savefig(outputFile)
plt.close()

#############################
#writing experience info txt#
#############################

now = datetime.now()

#writing experience info txt
lines = ['Experiment info',
         '',
         'Date and time: '+now.strftime("%d/%m/%Y %H:%M:%S"),
         'Device: TiePie',
         'Task: SNR',
         'Samp. freq.: '+str(f),
         'acq. time.: '+str(acqTime),
         'Num. traces: '+str(N),
         'Coupling: '+coupling,
         'Description: '+experimentDescription,
         'SNR Floor: '+str(SNR_floor)+'db',
         'SNR Laser: '+str(SNR_laser)+'db',
         ]

with open(os.path.join(outputFolder,'experimentInfo.txt'), 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')
