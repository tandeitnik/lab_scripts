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
minVoltage = -5 #TiePie minimum is -12V
maxVoltage = 5 #TiePie maximum is +12V
reps = 20 #number of different voltages the data is collected - collect ascending
auto = 0 #if auto = 1, the voltages are applied automatically, else the user must apply voltages via an external function generator and inform the entered values

#oscilloscope setup
#####################

f = int(1e6) #sampling frequency [Hz]
acqTime = 3 # total acquisiton time [s]
rootFolder = r"C:\Users\Labq\Desktop\Daniel R.T\Nova pasta\Nova pasta" #output folder where the calibration data will be saved
channel = 'ch1'

#write a description of the experiment
experimentDescription = "Electrode calibration via harmonic driving force. Vacumm chamber at XXmbar."


#calibration parameters
#####################

calibrationFactor = ufloat(26660.711505787207, 1137.7450977944743) #detector calibration factor [V/m] got from the detectorCalibration script. Pass as a ufloat variable.
omega = 2*np.pi*ufloat(80_000,5_000) #enter the frequency found in the detector calibration procedure


#physics setup
#####################

rho = 2200 #[kg / m3] particle density
diameter = ufloat(143e-9 ,  0.004e-6) #[m] particle diameter


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
    ch.range = 12
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

gen.signal_type = libtiepie.ST_DC

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

repsNumberList = generateOrderedNumbers(reps)

folders = next(os.walk(rootFolder))[1]
folderNumberList = generateOrderedNumbers(len(folders)+1)

for i in range(len(folders)+1):
    
    if ("elecCalibDataDC_"+folderNumberList[i]) in folders:
        pass
    else:
        outputFolder = os.path.join(rootFolder,"elecCalibDataDC_"+folderNumberList[i])
        os.mkdir(outputFolder)
        break

voltageValues = np.linspace(minVoltage,maxVoltage,reps)
meanPosition = unumpy.uarray([0]*reps,[0]*reps)

print("acquiring data")

for rep in range(reps):
    
    if auto == 0:
        
        winsound.Beep (440, 1000)
        #getting voltage from user
        voltageValues[rep] = float(input("\nType applyed voltage amplitude (peak to peak): "))
        input("\nPress ENTER to get next set of measurements...")
        
    else:
        
        print(str(rep+1)+"/"+str(reps))
    
    #setting and turning the waveform generator on
    gen.amplitude = voltageValues[rep]
    gen.start()
    
    #getting oscilloscope range 
    scp.start()

    # Wait for measurement to complete:
    while not (scp.is_data_ready or scp.is_data_overflow):
        time.sleep(0.01)  # 10 ms delay, to save CPU time

    # Get data:
    data = scp.get_data()
    

    # Stop stream:
    scp.stop()
    
    if channel == 'ch1':
        rangeMean = np.mean(data[0])
        rangeStd  = np.std(data[0])
    elif channel == 'ch2':
        rangeMean = np.mean(data[1])
        rangeStd  = np.std(data[1])
    
    autoRangeValue = rangeMean+3*rangeStd
    ch.range = autoRangeValue
    
    #getting data
 

    # Start measurement:
    scp.start()

    # Wait for measurement to complete:
    while not (scp.is_data_ready or scp.is_data_overflow):
        time.sleep(0.01)  # 10 ms delay, to save CPU time

    # Get data:
    data = scp.get_data()
    
    # Stop stream:
    scp.stop()
    
    #turn off waveform generator
    gen.stop()
    
    # evaluate mean and std
    if channel == 'ch1':
        
        meanPosition[rep] = ufloat(np.mean(data[0]) , np.std(data[0]))/calibrationFactor

    elif channel == 'ch2':

        meanPosition[rep] = ufloat(np.mean(data[1]) , np.std(data[1]))/calibrationFactor
    
    del data
    
# Close oscilloscope and generator:
# Disable output:
gen.output_on = False
del scp, gen
    
#putting applied voltage and position into a data frame
df = pd.DataFrame({'voltage [V]':voltageValues, '<position> [m]':meanPosition})
#saving data frame
outputFile = os.path.join(outputFolder,'experimentalData.pkl')
df.to_pickle(outputFile)

##################################################
#MAKING LINEAR REGRESSION OF FORCE VERSUS VOLTAGE#
##################################################

def linearRegression(x,a,b):
    
    return a*x+b

#discovering hints
hint_a = (unumpy.nominal_values(df['<position> [m]'][len(df)-1]) - unumpy.nominal_values(df['<position> [m]'][0]))/(unumpy.nominal_values(df['voltage [V]'][len(df)-1]) - unumpy.nominal_values(df['voltage [V]'][0]))
hint_b = unumpy.nominal_values(df['<position> [m]'][len(df)-1]) - hint_a*unumpy.nominal_values(df['voltage [V]'][len(df)-1])
#making fit
fit = curve_fit(linearRegression, df['voltage [V]'], unumpy.nominal_values(df['<position> [m]']) , p0 = [hint_a,hint_b], sigma= unumpy.std_devs(df['<position> [m]']), absolute_sigma=True)
ans, cov = fit
#evaluating calibration factor
slope = ufloat(ans[0] , np.sqrt(cov[0,0]) )
volume = (4/3)*np.pi*(diameter/2)**3 #[m**3]
mass = volume*rho #[kg]
elecCalibFactor =  mass*omega**2*slope #[N/V]

print("\nThe calibration factor {:.2u}" .format(elecCalibFactor*1e9) + "[pN/mV]")

########################################
#saving results and showing to the user#
########################################

#showing and saving a plot

dpi = 100
fig = plt.figure(figsize=(1.5*1080/dpi,1.5*720/dpi), dpi=dpi)
plt.rcParams.update({'font.size': 24})
plt.rcParams["axes.linewidth"] = 1

ax = plt.gca()

ax.errorbar(df['voltage [V]'], unumpy.nominal_values(df['<position> [m]'])*1e9, yerr=unumpy.std_devs(df['<position> [m]'])*1e9, xerr=None , fmt='o', label = 'measured mean position')
ax.plot(df['voltage [V]'],linearRegression(df['voltage [V]'],ans[0],ans[1])*1e9, label = 'linear fit')

ax.legend()
ax.set(ylabel=r'$\langle z \rangle$ [nm]')
ax.set(xlabel=r'voltage [V]')
ax.grid(alpha = 0.4)
plt.tight_layout()

outputFile = os.path.join(outputFolder,'positionVSvoltageFIT.png')
plt.savefig(outputFile)


#############################
#writing experience info txt#
#############################

now = datetime.now()

#writing experience info txt
lines = ['Experiment info',
         '',
         'Date and time: '+now.strftime("%d/%m/%Y %H:%M:%S"),
         'Device: TiePie',
         'Device: Electrode calibration - DC method',
         'Min. Voltage: '+str(minVoltage),
         'Max. Voltage: '+str(maxVoltage),
         'Auto/Manual: '+ str(auto),
         'Samp. freq.: '+str(f),
         'acq. time.: '+str(acqTime),
         'Description: '+experimentDescription,
         "\nThe calibration factor {:.2u}" .format(elecCalibFactor*1e9) + "[pN/mV]"
         ]

with open(os.path.join(outputFolder,'experimentInfo.txt'), 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')
