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
import winsound

####################
#PARAMETERS SECTION#
####################

#experiment setup
#####################
minVoltage = 1 #TiePie minimum is -12V
maxVoltage = 10 #TiePie maximum is +12V
reps = 10 #number of different voltages the data is collected - collect ascending
drivingFreq = 55_000 #driving harmonic frequency applyed
freqRange = 100 #frequency window used arount the drivingFreq
auto = 1 #if auto = 1, the voltages are applied automatically, else the user must apply voltages via an external function generator and inform the entered values


#oscilloscope setup
#####################

f = int(1e6) #sampling frequency [Hz]
acqTime = 0.1 # total acquisiton time [s]
N = 500 # number of traces
rootFolder = r"C:\Users\Labq\Desktop\Daniel R.T\TiePie\testes" #output folder where the calibration data will be saved
coupling= "ACV" #coupling type, can be ACV or DCV.
voltageRange = 1e-3 #oscilloscope range
autoRange = 1 #if it equals to 1, voltageRange will be ignored and an automatic range will be determined
gainAutoRange = 3 #multiplicative factor that determines the autoRange

#write a description of the experiment
experimentDescription = "Electrode calibration via harmonic driving force. Vacumm chamber at 10mbar."


#PSD setup
#####################

windows = 1 #number of windows used for the welch method
channel = 'ch1' #which channel to evaluate PSD, can be 'ch1' or 'ch2'
welchMethod = 1 #if welchMethod == 1, then Welch method is used (which is quicker but it is an estimation). Otherwise, periodogram is used.


#calibration parameters
#####################

calibrationFactor = ufloat(8847.800695841712, 418.144454770108) #detector calibration factor [V/m] got from the detectorCalibration script. Pass as a ufloat variable.
leftCut = 40_000  #left frequency cut - try to use the same as the one used for the detector calibration
rightCut = 65_000 #right frequency cut - try to use the same as the one used for the detector calibration
hint = [277598,39097,52200,0] #hints used to fit the lorentzian - use the same hints discovered at the detector calibration

#physics setup
#####################

kb = 1.380649e-23 # [m2 kg s-2 K-1]
T = ufloat(293.15, 1) #[K]
rho = 2200 #[kg / m3]
diameter = ufloat(143e-9 ,  0.004e-6) #[m]


######################
#connecting to TiePie#
######################

#Oscilloscope part

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


#Waveform generation part
    
# Try to open a generator:
gen = None
for item in libtiepie.device_list:
    if item.can_open(libtiepie.DEVICETYPE_GENERATOR):
        gen = item.open_generator()
        if gen:
            break
        
assert gen != None, "COULDN'T START THE WAVEFORM GENERATOR"

gen.signal_type = libtiepie.ST_SINE

# Set frequency:
gen.frequency = drivingFreq

# Set offset:
gen.offset = 0  # 0 V

# Set amplitude:
gen.amplitude = 0

# Enable output:
gen.output_on = True

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
repsNumberList = generateOrderedNumbers(reps)


folders = next(os.walk(rootFolder))[1]
folderNumberList = generateOrderedNumbers(len(folders)+1)

for i in range(len(folders)+1):
    
    if ("elecCalibDataAC_"+folderNumberList[i]) in folders:
        pass
    else:
        outputFolder = os.path.join(rootFolder,"elecCalibDataAC_"+folderNumberList[i])
        os.mkdir(outputFolder)
        break

voltageValues = np.linspace(minVoltage,maxVoltage,reps)
PSDList = []
trimmedPSD = []

print("acquiring data")

for rep in range(reps):
    
    if auto == 0:
        
        winsound.Beep (440, 1000)
        #getting voltage from user
        voltageValues[rep] = float(input("\nType applyed voltage amplitude (peak to peak): "))
        input("\nPress ENTER to get next set of measurements...")
        
    else:
        
        print(str(rep+1)+"/"+str(reps))

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
    
    #setting and turning the waveform generator on
    gen.amplitude = voltageValues[rep]
    gen.start()
    
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
        df = pd.DataFrame({'t':np.linspace(0,acqTime,size), 'ch1':data[0], 'ch2':data[1]})
        
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
    
    #turn off waveform generator
    gen.stop()
    
    #evaluates final PSD and stores it
    PSD_voltage = unumpy.uarray( np.mean(powerArray, axis = 0) , np.std(powerArray,axis = 0) )
    PSD_meters = PSD_voltage/(calibrationFactor**2)
    
    PSD = pd.DataFrame( { 'f [Hz]':freq, 'power [m**2/Hz]':PSD_meters})
    PSDList.append(PSD)
    
    #trimming
    deltaFreq = freq[1]-freq[0]
    idxLeft = int(leftCut/deltaFreq)
    idxRight = int(rightCut/deltaFreq)
    
    trPSD = PSD[idxLeft:idxRight].reset_index()
    trimmedPSD.append(trPSD)
   
    #saving stuff
    outputFile = os.path.join(outputFolder,'PSD.pkl')
    PSD.to_pickle(outputFile)
    outputFile = os.path.join(outputFolder,'PSDtrimmed.pkl')
    trPSD.to_pickle(outputFile)
    
    #delete original df to save space
    del df, powerArray, PSD, trPSD

# Close oscilloscope and generator:
# Disable output:
gen.output_on = False
del scp, gen

##############################################
#discovering the peak height for each voltage#
##############################################

def modelSimplified(f,D,gamma,f_0,cst):
    
    numerator = D*gamma
    w = f*2*np.pi
    w_0 = f_0*2*np.pi
    denominator = (w**2-w_0**2)**2 +(gamma*w)**2
    
    return  numerator/denominator + cst

elecForce = unumpy.uarray([0]*reps,[0]*reps)

#calculating mass with error
volume = (4/3)*np.pi*(diameter/2)**3 #[m**3]
mass = volume*rho #[kg]

for volt in range(reps):

    #first fitting a lorentzian to the trimmedPSD
    fit = curve_fit(modelSimplified,trimmedPSD[volt]['f [Hz]'],unumpy.nominal_values(trimmedPSD[volt]['power [m**2/Hz]']), p0 = hint, sigma= unumpy.std_devs(trimmedPSD[volt]['power [m**2/Hz]']), absolute_sigma=True)
    ans, cov = fit
    
    #unpacking fit parameters and transforming them in floats with uncertainty
    D_fit = ufloat(ans[0], np.sqrt(cov[0][0]))
    gamma_fit = ufloat(ans[1], np.sqrt(cov[1][1]))
    f_0_fit = ufloat(ans[2], np.sqrt(cov[2][2]))
    cst_fit = ufloat(ans[3], np.sqrt(cov[3][3]))
    
    #PSD value at the driving frequency
    S_drive = modelSimplified(drivingFreq,D_fit,gamma_fit,f_0_fit,cst_fit)
    
    #discovering the top height
    deltaFreq = trimmedPSD[volt]['f [Hz]'][1] -trimmedPSD[volt]['f [Hz]'][0]
    idxCentral = int((drivingFreq-leftCut)/deltaFreq)
    idxLeft = int(idxCentral-(freqRange/deltaFreq))
    idxRight = int(idxCentral+(freqRange/deltaFreq))
    
    top = max(unumpy.nominal_values(trimmedPSD[volt]['power [m**2/Hz]'][idxLeft:idxRight]))
    
    h = top-S_drive
    
    #calculating force
    tau = acqTime/2
    w0 = 2*np.pi*f_0_fit
    wDrive = 2*np.pi*drivingFreq
    DeltaDrive = w0**2 - wDrive**2
    
    force = sqrt( (h/tau)*mass**2*(DeltaDrive**2 + gamma_fit**2*wDrive**2) )
    elecForce[volt] = force
    
    #plt.plot(trimmedPSD[volt]['f [Hz]'],unumpy.nominal_values(trimmedPSD[volt]['power [m**2/Hz]']))
    #plt.plot(trimmedPSD[volt]['f [Hz]'],modelSimplified(trimmedPSD[volt]['f [Hz]'],ans[0],ans[1],ans[2],ans[3]))

    
#putting results in a data frame and saving
df = pd.DataFrame({'voltage [V]':voltageValues , 'electric force [N]':elecForce})
outputFile = os.path.join(outputFolder,'voltageVSforce.pkl')
df.to_pickle(outputFile)

#plt.plot(df['voltage [V]'],unumpy.nominal_values(df['electric force [N]']))

##################################################
#MAKING LINEAR REGRESSION OF FORCE VERSUS VOLTAGE#
##################################################

def linearRegression(x,a,b):
    
    return a*x+b

#discovering hints
hint_a = (unumpy.nominal_values(df['electric force [N]'][len(df)-1]) - unumpy.nominal_values(df['electric force [N]'][0]))/(unumpy.nominal_values(df['voltage [V]'][len(df)-1]) - unumpy.nominal_values(df['voltage [V]'][0]))
hint_b = unumpy.nominal_values(df['electric force [N]'][len(df)-1]) - hint_a*unumpy.nominal_values(df['voltage [V]'][len(df)-1])

fit = curve_fit(linearRegression, df['voltage [V]'], unumpy.nominal_values(df['electric force [N]']) , p0 = [hint_a,hint_b], sigma= unumpy.std_devs(df['electric force [N]']), absolute_sigma=True)
ans, cov = fit

elecCalibFactor = ufloat(ans[0] , np.sqrt(cov[0,0]) ) #[N/V]


########################################
#saving results and showing to the user#
########################################

#showing and saving a plot

dpi = 100
fig = plt.figure(figsize=(1.5*1080/dpi,1.5*720/dpi), dpi=dpi)
plt.rcParams.update({'font.size': 24})
plt.rcParams["axes.linewidth"] = 1

ax = plt.gca()

ax.errorbar(df['voltage [V]'], unumpy.nominal_values(df['electric force [N]'])*1e12, yerr=unumpy.std_devs(df['electric force [N]'])*1e12, xerr=None , fmt='o', label = 'measured electric force')
ax.plot(df['voltage [V]'],linearRegression(df['voltage [V]'],ans[0],ans[1])*1e12, label = 'linear fit')

ax.legend()
ax.set(ylabel=r'$F_{el}$ [pN]')
ax.set(xlabel=r'voltage [V]')
ax.grid(alpha = 0.4)
plt.tight_layout()

outputFile = os.path.join(outputFolder,'forceVSvoltageFIT.png')
plt.savefig(outputFile)
print("\nThe calibration factor {:.2u}" .format(elecCalibFactor*1e9) + "[pN/mV]")

#############################
#writing experience info txt#
#############################

now = datetime.now()

#writing experience info txt
lines = ['Experiment info',
         '',
         'Date and time: '+now.strftime("%d/%m/%Y %H:%M:%S"),
         'Device: TiePie',
         'Device: Electrode calibration - AC method',
         'Min. Voltage: '+str(minVoltage),
         'Max. Voltage: '+str(maxVoltage),
         'Driving freq.'+str(drivingFreq),
         'Auto/Manual: '+ str(auto),
         'Samp. freq.: '+str(f),
         'acq. time.: '+str(acqTime),
         'Num. traces: '+str(N),
         'Coupling: '+coupling,
         'Description: '+experimentDescription,
         "\nThe calibration factor {:.2u}" .format(elecCalibFactor*1e9) + "[pN/mV]"
         ]

with open(os.path.join(outputFolder,'experimentInfo.txt'), 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')
