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
from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties import unumpy
import beepy

####################
#PARAMETERS SECTION#
####################


#oscilloscope setup
#####################

f = int(1e6) #sampling frequency [Hz]
acqTime = 0.1 # total acquisiton time [s]
N = 1000 # number of traces
rootFolder = r"C:\Users\Labq\Desktop\Daniel R.T\Nova pasta\calibrationTest" #output folder where the calibration data will be saved
coupling= "ACV" #coupling type, can be ACV or DCV.
voltageRange = 1e-3 #oscilloscope range
autoRange = 1 #if it equals to 1, voltageRange will be ignored and an automatic range will be determined
gainAutoRange = 1.5 #multiplicative factor that determines the autoRange

#write a description of the experiment
experimentDescription = "Detector calibration. Vacumm chamber at XXmbar."

saveRawData = 0 #if 1 the raw data is saved, else the raw data is deleted and only the mean PSD is saved

#PSD setup
#####################

windows = 10 #number of windows used for the welch method
channel = 'ch1' #which channel to evaluate PSD, can be 'ch1' or 'ch2'
welchMethod = 1 #if welchMethod == 1, then Welch method is used (which is quicker but it is an estimation). Otherwise, periodogram is used.

#physics setup
#####################

kb = 1.380649e-23 # [m2 kg s-2 K-1]
T = ufloat(293.15, 1) #[K]
rho = 2200 #[kg / m3]
radius = ufloat(143e-9/2 , 10e-9) #[m]


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

# Close oscilloscope:
del scp

# Calculate mean PSD and standard error
PSD_voltage = unumpy.uarray( np.mean(powerArray, axis = 0) , np.std(powerArray,axis = 0) )
df_PSD = pd.DataFrame({'f [Hz]':freq, 'power [V**2/Hz]':PSD_voltage})

#saving the data frame with PSD
outputFile = os.path.join(outputFolder,'PSD.pkl')
df_PSD.to_pickle(outputFile)

del powerArray


#################################################
#getting left and right frequency cuts from user#
#################################################
        
#printing result so that the user may decide where to trimm the PSD
fig = plt.figure()
plt.rcParams.update({'font.size': 14})
plt.rcParams["axes.linewidth"] = 1

ax = plt.gca()

ax.scatter(df_PSD['f [Hz]'] ,unumpy.nominal_values(df_PSD['power [V**2/Hz]']), s = 10)
ax.set_ylim([min(unumpy.nominal_values(df_PSD['power [V**2/Hz]'][1:])), max(unumpy.nominal_values(df_PSD['power [V**2/Hz]']))])
ax.set_xlim([1000, f/2])
        
ax.set_yscale('log')
ax.set_xscale('log')
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

trimmedPSD = pd.DataFrame({'f [Hz]':df_PSD['f [Hz]'][idxLeft:idxRight], 'power [V**2/Hz]':df_PSD['power [V**2/Hz]'][idxLeft:idxRight]}).reset_index()
#saving the data frame with trimmed PSD
outputFile = os.path.join(outputFolder,'trimmedPSD.pkl')
trimmedPSD.to_pickle(outputFile)

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
Sm = unumpy.nominal_values(max(trimmedPSD['power [V**2/Hz]']))

#2) discover approximate frequency where the PSD is at half value
for i in range(len(trimmedPSD)):
    
    if unumpy.nominal_values(trimmedPSD['power [V**2/Hz]'][i]) >= Sm:
        f_0_hint  = trimmedPSD['f [Hz]'][i]
        break

for i in range(len(trimmedPSD)):

    if unumpy.nominal_values(trimmedPSD['power [V**2/Hz]'][i]) >= Sm/2 and (('f_l' in locals()) == False):
        f_l = trimmedPSD['f [Hz]'][i]
        
    if unumpy.nominal_values(trimmedPSD['power [V**2/Hz]'][i]) <= Sm/2 and (trimmedPSD['f [Hz]'][i] > f_0_hint):
        f_r = trimmedPSD['f [Hz]'][i]
        break
    
#3) evaluate the hints
w_0_hint = f_0_hint*2*np.pi
w_r = f_r*2*np.pi
w_l = f_l*np.pi*2

gamma_hint = np.sqrt( abs(((w_0_hint**2-w_l**2)**2 - (w_0_hint**2-w_r**2)**2) / (w_r**2 - w_l**2)))
D_hint = Sm*gamma_hint*w_0_hint**2
cst_hint = unumpy.nominal_values(np.min(trimmedPSD['power [V**2/Hz]']))
hint = [D_hint,gamma_hint,f_0_hint,cst_hint]

#fitting
fit = curve_fit(modelSimplified,trimmedPSD['f [Hz]'],unumpy.nominal_values(trimmedPSD['power [V**2/Hz]']), p0 = hint, sigma= unumpy.std_devs(trimmedPSD['power [V**2/Hz]']), absolute_sigma=True)
ans, cov = fit

##############################################################
#calculating calibration factor with appropiate uncertainties#
##############################################################

volume = (4/3)*np.pi*radius**3 #[m**3]
mass = volume*rho #[kg]

D = ufloat(ans[0] , np.sqrt(cov[0,0]))

calibrationFactor = sqrt(D*mass/(4*kb*T)) #[V/m]


########################################
#saving results and showing to the user#
########################################

#showing and saving a plot

fig = plt.figure()
plt.rcParams.update({'font.size': 14})
plt.rcParams["axes.linewidth"] = 1

ax = plt.gca()

ax.scatter(trimmedPSD['f [Hz]'],unumpy.nominal_values(trimmedPSD['power [V**2/Hz]']),label = 'trimmed PSD' , s = 10)
ax.plot(trimmedPSD['f [Hz]'],modelSimplified(trimmedPSD['f [Hz]'],ans[0],ans[1],ans[2],ans[3]), 'r',label='fitted function')

ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
ax.set(ylabel=r'$V^2$/Hz')
ax.set(xlabel=r'$\Omega/2\pi$ [Hz]')
ax.grid(alpha = 0.4)
plt.tight_layout()

outputFile = os.path.join(outputFolder,'trimmedPSDwithFIT.png')
plt.savefig(outputFile)
print('\nThe calibration factor is: {:.2u}'.format(calibrationFactor*1e-6)+ " [mV/nm]" )

#############################
#writing experience info txt#
#############################

now = datetime.now()

#writing experience info txt
lines = ['Experiment info',
         '',
         'Date and time: '+now.strftime("%d/%m/%Y %H:%M:%S"),
         'Device: TiePie',
         'Task: Detector calibration',
         'Samp. freq.: '+str(f),
         'acq. time.: '+str(acqTime),
         'Num. traces: '+str(N),
         'Coupling: '+coupling,
         'Description: '+experimentDescription,
         '\nThe calibration factor is: {:.2u}'.format(calibrationFactor*1e-6)+ " [mV/nm]"
         ]

with open(os.path.join(rootFolder,'experimentInfo.txt'), 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')
