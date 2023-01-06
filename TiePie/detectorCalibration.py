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
from scipy.optimize import curve_fit
import beepy

####################
#PARAMETERS SECTION#
####################

f = int(1e6) #sampling frequency [Hz]
acqTime = 0.04 # total acquisiton time [s]
N = 100 # number of traces
rootFolder = r"C:\Users\Labq\Desktop\Daniel R.T\Nova pasta\data\backward-noparticle" #output folder where the calibration data will be saved
coupling= "ACV" #coupling type, can be ACV or DCV.
voltageRange = 1e-3 #oscilloscope range
autoRange = 1 #if it equals to 1, voltageRange will be ignored and an automatic range will be determined
gainAutoRange = 1.5 #multiplicative factor that determines the autoRange

windows = 10 #number of windows used for the welch method
channel = 'ch1' #which channel to evaluate PSD, can be 'ch1' or 'ch2'
welchMethod = 1 #if welchMethod == 1, then Welch method is used (which is quicker but it is an estimation). Otherwise, periodogram is used.

kb = 1.380649e-23 # [m2 kg s-2 K-1]
T = 293.15 #[K]
rho = 2200 #[kg / m3]
radius = 143e-9/2 #[m]
volume = (4/3)*np.pi*radius**3
mass = volume*rho

experimentDescription = "Detector calibration. Vacumm chamber at XXmbar."


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
    #print("Range set to "+str(autoRangeValue)+"V")


####################
#acquiring the data#
####################

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

tracesNumberList = generateOrderedNumbers(N)

outputFolder = os.path.join(rootFolder,"calibrationData")
os.mkdir(outputFolder)

dataList = []

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

    #putdF into a list
    dataList.append(df)
    
    #delete original df to save space
    del df


#################
#saving the data#
#################

print("saving data")

for n in tqdm(range(N)):
    
    outputFile = os.path.join(outputFolder,tracesNumberList[n])
    dataList[n].to_pickle(outputFile)

# Close oscilloscope:
del scp


#####################
#evaluating full PSD#
#####################

print("calculating PSD")

powerList = []
stdList = []

#evaluating the PSD
if welchMethod == 1: #if welch method is ON
    
    #evaluates the PSD for the first trace
    freq, power = signal.welch(dataList[0][channel], f, window = 'hamming', nperseg = int(len(dataList[0][channel])/windows))
    powerArray = np.zeros([N,len(freq)])
    
    #evaluates the PSD for the subsequent traces and sum
    for i in range(1,len(data)):
        freq, powerTemp = signal.welch(dataList[i][channel], f, window = 'hamming', nperseg = int(len(dataList[i][channel])/windows))
        powerArray[i,:] = power
        
    
else: #if welch method is OFF
    
    #evaluates the PSD for the first trace
    freq, power = signal.periodogram(dataList[0][channel], f, scaling='density')
    powerArray = np.zeros([N,len(freq)])
    
    #evaluates the PSD for the subsequent traces and sum
    for i in range(1,len(data)):
        
        freq, powerTemp = signal.periodogram(dataList[i], f, scaling='density')
        powerArray[i,:] = power
        
PSD_mean = np.mean(powerArray, axis = 0)
PSD_std = np.std(powerArray,axis = 0)
del powerArray


#################################################
#getting left and right frequency cuts from user#
#################################################
        
#printing result so that the user may decide where to trimm the PSD
fig = plt.figure()
plt.rcParams.update({'font.size': 14})
plt.rcParams["axes.linewidth"] = 1

ax = plt.gca()

ax.scatter(freq ,PSD_mean, s = 10)
ax.set_xlim([1000, f/2])
        
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
ax.set(ylabel=r'$V^2$/Hz')
ax.set(xlabel=r'$\Omega/2\pi$ [Hz]')
ax.grid(alpha = 0.4)
plt.tight_layout()

outputFile = os.path.join(outputFolder,'fullPSD.png')
plt.savefig(outputFile)
plt.close()

beepy.beep(sound=1)
print("\nPlease, open the image file 'fullPSD.png' in "+outputFolder)
print("\nFrom the plot, you should select a minimum and maximum frequencies to cut the PSD.")

agreementTest = 0

while agreementTest == 0:
    
    leftCut = input("\nEnter a frequency value for the minimum frequency and press ENTER: ")
    rightCut = input("\nEnter a frequency value for the maximum frequency and press ENTER: ")
    print("\nYou entered "+leftCut+"Hz for the minimum frequency and "+rightCut+"Hz for the maximum frequency.")
    agree = input("Do you agree? [y/n]")
    
    while (agree != 'y') and  (agree != 'n'):
        agree = input("\ninvalid input, please enter y if you agree or n if you disagree and want to replace the values: ")

    if agree == 'y':
        agreementTest = 1
    
#trimming the PSD

deltaFreq = freq[1]-freq[0]
idxLeft = int(leftCut/deltaFreq)
idxRight = int(rightCut/deltaFreq)

trimmedFreq = freq[idxLeft:idxRight]
trimmedPSD  = PSD_mean[idxLeft:idxRight]
trimmedSTD  = PSD_std[idxLeft:idxRight]


################
#making the fit#
################

def modelSimplified(f,D,gamma,f_0,cst):
    
    numerator = D*gamma
    w = f*2*np.pi
    w_0 = f_0*2*np.pi
    denominator = (w**2-w_0**2)**2 +(gamma*w)**2
    
    return  numerator/denominator + cst
    
#discovering hints for fit

#1) discover max value of the PSD
Sm = np.max(trimmedPSD)

#2) discover approximate frequency where the PSD is at half value
for i in range(len(trimmedPSD)):
    
    if trimmedPSD[i] >= Sm:
        f_0_hint  = trimmedFreq[i]
        break

for i in range(len(trimmedPSD)):

    if trimmedPSD[i] >= Sm/2 and (('f_l' in locals()) == False):
        f_l = trimmedFreq[i]
        
    if trimmedPSD[i] <= Sm/2 and (trimmedFreq[i] > f_0_hint):
        f_r = trimmedFreq[i]
        break
    
#3) evaluate the hints
w_0_hint = f_0_hint*2*np.pi
w_r = f_r*2*np.pi
w_l = f_l*np.pi

gamma_hint = np.sqrt( ((w_0_hint**2-w_l**2)**2 - (w_0_hint**2-w_r**2)**2) / (w_r**2 - w_l**2))
D_hint = Sm*gamma_hint*w_0_hint**2
cst_hint = np.min(trimmedPSD)
hint = [D_hint,gamma_hint,f_0_hint,cst_hint]

#fitting
fit = curve_fit(modelSimplified,trimmedFreq,trimmedPSD, p0 = hint, sigma= trimmedSTD, absolute_sigma=True)

#calculating calibration factor
ans, cov = fit
calibrationFactor = np.sqrt(ans[0]*np.pi*mass/(2*kb*T)) #[V/m]
#STILL LACKS THE UNCERTAINTY!!!!
error = 1


########################################
#saving results and showing to the user#
########################################

#showing and saving a plot

fig = plt.figure()
plt.rcParams.update({'font.size': 14})
plt.rcParams["axes.linewidth"] = 1

ax = plt.gca()

ax.scatter(trimmedFreq,trimmedPSD,label = 'trimmed PSD' , s = 10)
ax.plot(trimmedFreq,modelSimplified(trimmedFreq,ans[0],ans[1],ans[2],ans[3]), 'r',label='fitted function')

ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
ax.set(ylabel=r'$V^2$/Hz')
ax.set(xlabel=r'$\Omega/2\pi$ [Hz]')
ax.grid(alpha = 0.4)
plt.tight_layout()

outputFile = os.path.join(outputFolder,'trimmedPSDwithFIT.png')
plt.savefig(outputFile)
print("\nThe calibration factor is %.2f" % (calibrationFactor/1e6) + " +-  %.2f" % (error/1e6) + " [mV/nm]")

#############################
#writing experience info txt#
#############################

now = datetime.now()

#writing experience info txt
lines = ['Experiment info',
         '',
         'Date and time: '+now.strftime("%d/%m/%Y %H:%M:%S"),
         'Device: TiePie',
         'Device: Detector calibration',
         'Samp. freq.: '+str(f),
         'acq. time.: '+str(acqTime),
         'Num. traces: '+str(N),
         'Coupling: '+coupling,
         'Description: '+experimentDescription,
         "\nThe calibration factor is %.2f" % (calibrationFactor/1e6) + " +-  %.2f" % (error/1e6) + " [mV/nm]"
         ]

with open(os.path.join(rootFolder,'experimentInfo.txt'), 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')