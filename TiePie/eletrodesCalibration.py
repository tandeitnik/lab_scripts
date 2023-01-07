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

####################
#PARAMETERS SECTION#
####################

#experiment setup
#####################
reps = 20 #number of different voltages the data is collected - collect ascending
drivingFreq = 83e3 #driving harmonic frequency applyed
freqRange = 100 #frequency window used arount the drivingFreq


#oscilloscope setup
#####################

f = int(1e6) #sampling frequency [Hz]
acqTime = 0.04 # total acquisiton time [s]
N = 100 # number of traces
rootFolder = r"C:\Users\Labq\Desktop\Daniel R.T\Nova pasta\data\backward-noparticle" #output folder where the calibration data will be saved
coupling= "ACV" #coupling type, can be ACV or DCV.
voltageRange = 1e-3 #oscilloscope range
autoRange = 1 #if it equals to 1, voltageRange will be ignored and an automatic range will be determined
gainAutoRange = 1.5 #multiplicative factor that determines the autoRange

#write a description of the experiment
experimentDescription = "Electrode calibration via harmonic driving force. Vacumm chamber at XXmbar."


#PSD setup
#####################

windows = 10 #number of windows used for the welch method
channel = 'ch1' #which channel to evaluate PSD, can be 'ch1' or 'ch2'
welchMethod = 1 #if welchMethod == 1, then Welch method is used (which is quicker but it is an estimation). Otherwise, periodogram is used.


#calibration parameters
#####################

calibrationFactor = 1 #detector calibration factor [V/m]
leftCut = 70_000  #left frequency cut - try to use the same as the one used for the detector calibration
rightCut = 110_000 #right frequency cut - try to use the same as the one used for the detector calibration
hint = [D_hint,gamma_hint,f_0_hint,cst_hint] #hints used to fit the lorentzian - use the same hints discovered at the detector calibration

#physics setup
#####################

kb = 1.380649e-23 # [m2 kg s-2 K-1]
T = 293.15 #[K]
tempError = 1 #[K]
rho = 2200 #[kg / m3]
radius = 143e-9/2 #[m]
radiusError = 10e-9 #[m]
volume = (4/3)*np.pi*radius**3 #[m**3]
mass = volume*rho #[kg]

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
repsNumberList = generateOrderedNumbers(reps)

outputFolder = os.path.join(rootFolder,"electrodesCalibrationData")
os.mkdir(outputFolder)

voltageValues = np.zeros(reps)
PSDList = []
trimmedPSD = []

print("acquiring data")

for rep in range(reps):
    
    #getting voltage from user
    voltageValues[rep] = float(input("Type applyed voltage amplitude (peak to peak): "))
    input("Press ENTER to get next set of measurements...")
    
    #creating folder
    outputFolder = os.path.join(rootFolder,"rep"+repsNumberList[rep])    
    os.mkdir(outputFolder)

    #getting autoRange if on
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
    
    #getting data and calculating PSD
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
        df = pd.DataFrame({'t':np.linspace(0,acqTime,size), 'ch1':data[0]/calibrationFactor, 'ch2':data[1]/calibrationFactor})
        
        if n == 0: #first run
        
            #evaluates the PSD for the first trace
            if welchMethod == 1:
                freq, power = signal.welch(df[channel], f, window = 'hamming', nperseg = int(len(df[channel])/windows))
            else:
                freq, power = signal.periodogram(df[channel], f, scaling='density')
            
            powerArray = np.zeros([N,len(freq)])
            powerArray[n,:] = power
            
        #evaluates the PSD for the subsequent traces and sum
        else:
            
            if welchMethod == 1:
                freq, power = signal.welch(df[channel], f, window = 'hamming', nperseg = int(len(df[channel])/windows))
            else:
                freq, power = signal.periodogram(df[channel], f, scaling='density')
            
            powerArray[n,:] = power
    
    #evaluates final PSD and stores it
    PSD = pd.DataFrame({'f [Hz]':freq, 'power [m**2/Hz]':np.mean(powerArray, axis = 0), 'std [m**2/Hz]':np.std(powerArray,axis = 0)})
    PSDList.append(PSD)
    
    #trimming
    deltaFreq = freq[1]-freq[0]
    idxLeft = int(leftCut/deltaFreq)
    idxRight = int(rightCut/deltaFreq)
    trPSD = pd.DataFrame({'f [Hz]':PSD['f [Hz]'][idxLeft:idxRight], 'power [m**2/Hz]':PSD['power [m**2/Hz]'][idxLeft:idxRight], 'std [m**2/Hz]':PSD['std [m**2/Hz]'][idxLeft:idxRight]})
    trimmedPSD.append(trPSD)
    
    #saving stuff
    outputFile = os.path.join(outputFolder,'PSD.pkl')
    PSD.to_pickle(outputFile)
    outputFile = os.path.join(outputFolder,'PSDtrimmed.pkl')
    trPSD.to_pickle(outputFile)
    
    #delete original df to save space
    del df, powerArray, PSD, trPSD

# Close oscilloscope:
del scp

##############################################
#discovering the peak height for each voltage#
##############################################

def modelSimplified(f,D,gamma,f_0,cst):
    
    numerator = D*gamma
    w = f*2*np.pi
    w_0 = f_0*2*np.pi
    denominator = (w**2-w_0**2)**2 +(gamma*w)**2
    
    return  numerator/denominator + cst

elecForce = np.zeros(reps)
elecForceError = np.zeros(reps)
ansList = []
covList = []

for volt in range(reps):
    
    #first fitting a lorentzian to the trimmedPSD
    fit = curve_fit(modelSimplified,trimmedPSD[volt]['f [Hz]'],trimmedPSD[volt]['power [m**2/Hz]'], p0 = hint, sigma= trimmedPSD[volt]['std [m**2/Hz]'], absolute_sigma=True)
    ans, cov = fit
    ansList.append(ans)
    covList.append(cov)
    
    #PSD value at the drivint frequency
    S_drive = modelSimplified(drivingFreq,ans[0],ans[1],ans[2],ans[3])
    
    #discovering the top height
    deltaFreq = trimmedPSD[volt]['f [Hz]'][1] -trimmedPSD[volt]['f [Hz]'][0]
    idxCentral = int(drivingFreq/deltaFreq)
    idxLeft = int(idxCentral-(freqRange/deltaFreq))
    idxRight = int(idxCentral+(freqRange/deltaFreq))
    
    top = max(trimmedPSD[volt]['power [m**2/Hz]'][idxLeft:idxRight])
    #getting index of top value, it will be used later
    for i in range(len(trimmedPSD[volt]['power [m**2/Hz]'][idxLeft:idxRight])):
        
        if trimmedPSD[volt]['power [m**2/Hz]'][idxLeft+i] == top:
            
            idxTop = idxLeft+i
            break

    h = top-S_drive
    
    #calculating force
    tau = acqTime/2
    w0 = 2*np.pi*ans[2]
    wDrive = 2*np.pi*drivingFreq
    DeltaDrive = w0**2 - wDrive**2
    gamma = ans[1]
    A = (h/tau)*mass**2*(DeltaDrive**2 + gamma**2*wDrive**2)
    
    force = np.sqrt( A )
    elecForce[volt] = force

    #calculating error
    
    #1) partial derivatives
    dFdh = 0.5*force/h
    dFdm = force/mass
    dFdgamma = A**(-0.5)*gamma*wDrive**2*mass**2*h/tau
    dFdw_0 = A**(-0.5)*mass**2*h/tau*(-2*w0)
    
    #2) errors of parameters
    #a) height error is more envolved, additional partial derivatives must be evaluated
    #calculating partial derivatives to evaluate error of the S_drive = b (b of bottom)
    D = ans[0]
    dbdD = S_drive/D
    dbdgamma = (D*(DeltaDrive**2+gamma**2*wDrive**2) - 2*D*gamma**2*wDrive**2) / (DeltaDrive**2+gamma**2*wDrive**2)**2
    dbdw_0 = (-1*D*gamma*4*DeltaDrive*w0) / (DeltaDrive**2+gamma**2*wDrive**2)**2
    
    deltab = np.sqrt( dbdD**2*cov[0,0] + dbdgamma**2*cov[1,1] + dbdw_0*cov[1,1]*(2*np.pi)**2)
    deltaTop = trimmedPSD[volt]['std [m**2/Hz]'][idxTop]
    
    deltaHeight = np.sqrt(deltab**2 + deltaTop**2)
    
    #error of the mass
    deltaMass = rho*4*np.pi*radius**2*radiusError
    
    #geathering all together to evaluate the error of the force
    elecForceError[volt] =  np.sqrt(dFdh**2*deltaHeight**2 + dFdm**2*deltaMass**2 + dFdgamma**2*cov[1,1] + dFdw_0**2*cov[1,1]*(2*np.pi)**2 ) 

#putting voltages, force and erros in a data frame

df = pd.DataFrame({'voltage [V]':voltageValues , 'electric force [N]':elecForce , 'error [N]':elecForceError})
outputFile = os.path.join(outputFolder,'voltageVSforce.pkl')
df.to_pickle(outputFile)

##################################################
#MAKING LINEAR REGRESSION OF FORCE VERSUS VOLTAGE#
##################################################

def linearRegression(x,a,b):
    
    return a*x+b

#discovering hints
hint_a = (df['electric force [N]'][-1] - df['electric force [N]'][0])/(df['voltage [V]'][-1] - df['voltage [V]'][0])
hint_b = df['electric force [N]'][-1] - hint_a*df['voltage [V]'][-1]


fit = curve_fit(linearRegression, df['voltage [V]'], df['electric force [N]'] , p0 = [hint_a,hint_b], sigma= df['error [N]'], absolute_sigma=True)
ans, cov = fit

electricalCalibrationFactor = ans[0]
error = np.sqrt(cov[0,0])

########################################
#saving results and showing to the user#
########################################

#showing and saving a plot

fig = plt.figure()
plt.rcParams.update({'font.size': 14})
plt.rcParams["axes.linewidth"] = 1

ax = plt.gca()

ax.errorbar(df['voltage [V]'], df['electric force [N]']*1e12, yerr=df['error [N]']*1e12, xerr=None , label = 'measured electric force')
ax.plot(df['voltage [V]'],linearRegression(df['voltage [V]'],ans[0],ans[1])*1e12, label = 'linear fit')

ax.legend()
ax.set(ylabel=r'$F_{el}$ [pN]')
ax.set(xlabel=r'$voltage [V]')
ax.grid(alpha = 0.4)
plt.tight_layout()

outputFile = os.path.join(outputFolder,'forceVSvoltageFIT.png')
plt.savefig(outputFile)
print("\nThe calibration factor is %f" % (electricalCalibrationFactor*1e9) + " +-  %f" % (error*1e9) + " [pN/mV]")

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
         'The calibration factor is %f' % (electricalCalibrationFactor*1e9) + " +-  %f" % (error*1e9) + " [pN/mV]"
         ]

with open(os.path.join(rootFolder,'experimentInfo.txt'), 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')