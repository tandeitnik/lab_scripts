# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 13:19:39 2023

@author: tandeitnik
"""

from PyQt5 import uic, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication,  QMessageBox
import sys
from __future__ import print_function
import time
import os
import libtiepie
import numpy as np
from printinfo import *
from datetime import datetime
import pandas as pd
import winsound
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties import unumpy

class fileName:
    
    def __init__(self, maxNumber):
        
        self.maxNumber = maxNumber
    
    def __iter__(self):
        
        n0 = int(np.log10(self.maxNumber))
        self.fileName = ['0']*(n0+1)
        return self
    
    def __next__(self):
        current = self.fileName
        carry = 1
        for i, digit in enumerate(current):
            if int(digit)+carry == 10:
                self.fileName[i] = '0'
                carry = 1
            else:
                self.fileName[i] = str(int(digit)+carry)
                break
        
        output = current[-1]
        for i in range(1,len(current)):
            output = output + current[-1-i]
            
        if int(output) == self.maxNumber:
            self.fileName = ['0']*(int(np.log10(self.maxNumber))+1)
            
        return output

def openHelpWindow():
    
    helpWindow.show()

def getFolderName():
    
    global rootFolder
    
    rootFolder = QtWidgets.QFileDialog.getExistingDirectory()
    mainWindow.rootFolderLineEdit_A.setText(rootFolder)
    mainWindow.rootFolderLineEdit_B_2.setText(rootFolder)
    
def alertaRootFolder():
    QMessageBox.warning(mainWindow,'Warning','Root folder path not valid.')
    
def alertaAcqTime():
    QMessageBox.warning(mainWindow,'Warning','Acquisition time can not be zero.')
    
def alertaVoltRange():
    QMessageBox.warning(mainWindow,'Warning','Voltage range can not be zero while auto range is set to off.')
    
def alertaTiePieNotFound():
    QMessageBox.warning(mainWindow,'Warning','Oscilloscope not found!')
    
def alertaProcessFinished():
    QMessageBox.warning(mainWindow,'Message','The process is over!')
    
def alertaLeftRight():
    QMessageBox.warning(mainWindow,'Message','Right frequency must be greater than left frequency!')

def alertaDefineLeftRightCuts():
    QMessageBox.warning(mainWindow,'Message','Please, choose the left/right frequencies cut')

def alertaDetCalNotStarted():
    QMessageBox.warning(mainWindow,'Warning',"Detector calibration was not initiallized yet. Please, click the arrow and refer to the previous screen.")

def parametersWindow_detCal():
    mainWindow.stackedWidget.setCurrentIndex(0)
    
def leftRightWindow_detCal():
    mainWindow.stackedWidget.setCurrentIndex(1)
    
def resultsWindow_detCal():
    mainWindow.stackedWidget.setCurrentIndex(2)
    
def results2Window_detCal():
    mainWindow.stackedWidget.setCurrentIndex(3)


def getParametersAcq():
    
    global reps, N, delay, experimentDescription, freq, acqTime, coupling
    global voltageRange, autoRange, rootFolder
    
    #experiment parameters
    reps = mainWindow.repetitionsSpinBox_A.value() #repetitions
    N = mainWindow.tracesSpinBox_A.value() #number of traces
    delay = mainWindow.delaySpinBox_A.value() #delay between repetitions
    experimentDescription = mainWindow.expDescription_A.text()
    
    #oscilloscope parameters
    freq = mainWindow.frequencySpinBox_A.value() #sampling frequency
    acqTime = mainWindow.acqTime_A.value() #acquiring time
    coupling = mainWindow.couplingComboBox_A.currentText()
    voltageRange = mainWindow.voltageRangeSpinBox_A.value()
    autoRange = 1 if (mainWindow.autoRangeComboBox_A.currentText() == 'ON') else 0
    
    #salving folder
    rootFolder = mainWindow.rootFolderLineEdit_A.text()
    
def getParametersDetCal():
    
    global N, saveRawData, experimentDescription, f, acqTime, coupling, voltageRange
    global autoRange, rootFolder, windows, channel, welchMethod
    global kb, T, rho, diameter
    
    #experiment parameters
    N = mainWindow.tracesSpinBox_B_2.value() #number of traces
    saveRawData = mainWindow.saveRaw_B_2.currentText()
    experimentDescription = mainWindow.expDescription_B_2.text()
    
    #oscilloscope parameters
    f = mainWindow.frequencySpinBox_B_2.value() #sampling frequency
    acqTime = mainWindow.acqTime_B_2.value() #acquiring time
    coupling = mainWindow.couplingComboBox_B_2.currentText()
    voltageRange = mainWindow.voltageRangeSpinBox_B_2.value()
    autoRange = 1 if (mainWindow.autoRangeComboBox_B_2.currentText() == 'ON') else 0
    
    #salving folder
    rootFolder = mainWindow.rootFolderLineEdit_B_2.text()
    
    #PSD parameters
    windows = mainWindow.psdWindows_B_2.value() #number of windows used for the welch method
    channel = mainWindow.psdChannel_B_2.currentText() #which channel to evaluate PSD, can be 'ch1' or 'ch2'
    welchMethod = 1 if (mainWindow.psdMethod_B_2.currentText() == 'Welch') else 0 #if welchMethod == 1, then Welch method is used (which is quicker but it is an estimation). Otherwise, periodogram is used.
    
    #Physical parameters
    kb = 1.380649e-23 # [m2 kg s-2 K-1]
    T = mainWindow.temperature_B_2.value()
    rho = 2200 #[kg / m3]
    diameter = ufloat(mainWindow.diameter_B_2.value()*1e-9 ,mainWindow.diameterStd_B_2.value()*1e-9) #[m]
    
    
def startAcquisition():
    
    mainWindow.startAcqButton_A.setEnabled(False)
    
    #get parameters
    getParametersAcq()
    
    if rootFolder != "" and os.path.exists(rootFolder):
        
        if acqTime == 0:
        
            alertaAcqTime()
            mainWindow.startAcqButton_A.setEnabled(True)
            
        elif (autoRange == 0) and (voltageRange == 0):
         
            alertaVoltRange()
            mainWindow.startAcqButton_A.setEnabled(True)
            
        else:
            
            #connect to tiepie
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
                        
            if scp != None:
    
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
                        
            else:
                
                alertaTiePieNotFound()
                mainWindow.startAcqButton_A.setEnabled(True)
            
            if scp != None:
                
                #creating folder
                now = datetime.now()
                timeString = str(datetime.now().hour)+"_"+str(datetime.now().minute)+"h"
                outputFolder = os.path.join(rootFolder,"data_"+timeString)
                os.mkdir(outputFolder)
                
                
                #save experiment info
    
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
    
                with open(os.path.join(outputFolder,'experimentInfo.txt'), 'w') as f:
                    for line in lines:
                        f.write(line)
                        f.write('\n')
                
                repFolderNumberList = fileName(reps)
                repFolderNumbers = iter(repFolderNumberList)
                
                for rep in reps:
                    
                    #create repetition folder
                    outputFolderRep = os.path.join(outputFolder,"rep"+next(repFolderNumbers))
                    os.mkdir(outputFolderRep)
                    
                    #get range if auto range is ON
                    if autoRange == 1:
                        
                        #put ranges in maximum first
                        for ch in scp.channels:
                            ch.range = 12
                        
                        #determine new ranges
                        scp.start()
                    
                        # Wait for measurement to complete:
                        while not (scp.is_data_ready or scp.is_data_overflow):
                            time.sleep(0.01)  # 10 ms delay, to save CPU time
                    
                        # Get data:
                        data = scp.get_data()
                        # Stop stream:
                        scp.stop()
                        
                        #set new ranges
                        for i, ch in enumerate(scp.channels):
                            ch.range = np.std(data[i])*3
                    
                    
                    #get traces
                    
                    tracesNumberList = fileName(N)
                    tracesNumbers = iter(tracesNumberList)
                    
                    for n in N:
                        
                        #update progress text
                        mainWindow.progressStatus_A.setText('Getting trace '+str(n+1)+'/'+str(N)+' for repetition '+str(rep+1)+'/'+str(reps))
                        
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
                        
                        outputFile = os.path.join(outputFolderRep,next(tracesNumbers)+'.pkl')
                        df.to_pickle(outputFile)
                        
                        #delete original df to save space
                        del df
                        
                        #advance progress bar
                        mainWindow.progressBar_A.setValue(mainWindow.progressBar_A.value() + 100/(reps*N))
                        
                    if rep != reps-1:
                        mainWindow.progressStatus_A.setText('Repetition '+str(rep+1)+'/'+str(reps)+" completed, sleeping for "+str(delay)+" seconds")
                        time.sleep(delay)
                        
                # Close oscilloscope:
                del scp
                
                #alert user its over
                mainWindow.progressStatus_A.setText("The data acquisition is over!")
                alertaProcessFinished()
    else:
        alertaRootFolder()
        mainWindow.startAcqButton_A.setEnabled(True)
        
        

def startDetCal():
    
    mainWindow.startDetCal_B.setEnabled(False)
    
    #this function does the first part of the calibration, it get all the experimental data
    global df_PSD, outputFolder
    
    #get parameters
    getParametersDetCal()
    
    if rootFolder != "" and os.path.exists(rootFolder):
        
        if acqTime == 0:
        
            alertaAcqTime()
            mainWindow.startDetCal_B.setEnabled(True)
            
        elif (autoRange == 0) and (voltageRange == 0):
         
            alertaVoltRange()
            mainWindow.startDetCal_B.setEnabled(True)
            
        else:
            
            #connect to tiepie
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
                        
            if scp != None:
    
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
                        
            else:
                
                alertaTiePieNotFound()
                mainWindow.startDetCal_B.setEnabled(True)
                
            
            if scp != None:
                
                #creating folder
                now = datetime.now()
                timeString = str(datetime.now().hour)+"_"+str(datetime.now().minute)+"h"
                outputFolder = os.path.join(rootFolder,"detectorCalibrationData_"+timeString)
                os.mkdir(outputFolder)
                
                #get range if auto range is ON
                if autoRange == 1:
                    
                    #put ranges in maximum first
                    for ch in scp.channels:
                        ch.range = 12
                    
                    #determine new ranges
                    scp.start()
                
                    # Wait for measurement to complete:
                    while not (scp.is_data_ready or scp.is_data_overflow):
                        time.sleep(0.01)  # 10 ms delay, to save CPU time
                
                    # Get data:
                    data = scp.get_data()
                    # Stop stream:
                    scp.stop()
                    
                    #set new ranges
                    for i, ch in enumerate(scp.channels):
                        ch.range = np.std(data[i])*3
                
                #getting data
                if saveRawData == 1:
                    
                    outputFolderRawData = os.path.join(outputFolder,"rawData")
                    os.mkdir(outputFolderRawData)
                    
                    tracesNumberList = fileName(N)
                    tracesNumbers = iter(tracesNumberList)
                    
                for n in range(N):
                    
                    mainWindow.progressBar_B_2.setValue(mainWindow.progressBar_A.value() + 100/(N))
                    mainWindow.progressStatus_B_2.setText('Acquiring trace '+str(n)+'/'+str(N))
                    
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
                    
                    #save rawData if on
                    if saveRawData == 1:
                        
                        outputFile = os.path.join(outputFolderRawData, next(tracesNumbers) )
                        df.to_pickle(outputFile)
                        
                    
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
                
                #plot the PSD
                dpi = 100
                fig = plt.figure(figsize=(1180/dpi,480/dpi), dpi=dpi)
                plt.rcParams.update({'font.size': 16})
                plt.rcParams["axes.linewidth"] = 1

                ax = plt.gca()

                ax.scatter(df_PSD['f [Hz]'] ,unumpy.nominal_values(df_PSD['power [V**2/Hz]']), s = 10)
                ax.set_ylim([min(unumpy.nominal_values(df_PSD['power [V**2/Hz]'][1:])), max(unumpy.nominal_values(df_PSD['power [V**2/Hz]']))])
                ax.set_xlim([1000, f/2])
                        
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
                
                #placing plot
                mainWindow.plot_B_2.setPixmap(QtGui.QPixmap(outputFile))
                mainWindow.stackedWidget.setCurrentIndex(1)  
                mainWindow.progressStatus_B_2.setText("waiting for left/right frequency cuts...")
                alertaDefineLeftRightCuts()
                
    else:
        alertaRootFolder()
        mainWindow.startDetCal_B.setEnabled(True)
        
def getLeftRightCut():
    
    mainWindow.LRbutton.setEnabled(False)
    
    global leftCut, rightCut
    
    leftCut = mainWindow.leftCut_B.value()
    rightCut = mainWindow.rightCut_B.value()

    if leftCut >= rightCut:
        
        alertaLeftRight()
        mainWindow.LRbutton.setEnabled(True)
        
    else:
        
        mainWindow.progressStatus_B_2.setText("evaluating detector calibration factor")
        continueDetCal()
                
def continueDetCal():
    
    global f
    #trimming the PSD

    deltaFreq = df_PSD['f [Hz]'][1]- df_PSD['f [Hz]'][1][0]
    idxLeft = int(leftCut/deltaFreq)
    idxRight = int(rightCut/deltaFreq)

    trimmedPSD = df_PSD[idxLeft:idxRight].reset_index()

    #saving the data frame with trimmed PSD
    outputFile = os.path.join(outputFolder,'trimmedPSD.pkl')
    trimmedPSD.to_pickle(outputFile)

    ################
    #making the fit#
    ################
    
    try:
        
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
        hint = [D_hint,gamma_hint,f_0_hint,0]
        
        mainWindow.label_43.setText(str(D_hint))
        mainWindow.label_102.setText(str(gamma_hint))
        mainWindow.label_109.setText(str(f_0_hint))
        
    
        #fitting
        fit = curve_fit(modelSimplified,trimmedPSD['f [Hz]'],unumpy.nominal_values(trimmedPSD['power [V**2/Hz]']), p0 = hint, sigma= unumpy.std_devs(trimmedPSD['power [V**2/Hz]']), absolute_sigma=True)
        ans, cov = fit
        
        mainWindow.label_103.setText(str(ans[0]))
        mainWindow.label_104.setText(str(np.sqrt(cov[0,0])))
        
        mainWindow.label_105.setText(str(ans[1]))
        mainWindow.label_107.setText(str(np.sqrt(cov[1,1])))
        
        mainWindow.label_106.setText(str(ans[2]))
        mainWindow.label_108.setText(str(np.sqrt(cov[2,2])))
        
        
    
        ##############################################################
        #calculating calibration factor with appropiate uncertainties#
        ##############################################################
    
        volume = (4/3)*np.pi*(diameter/2)**3 #[m**3]
        mass = volume*rho #[kg]
    
        D = ufloat(ans[0] , np.sqrt(cov[0,0]))
    
        calibrationFactor = sqrt(D*mass/(4*kb*T)) #[V/m]
        
        ########################################
        #saving results and showing to the user#
        ########################################
    
        #showing and saving a plot
    
        dpi = 100
        fig = plt.figure(figsize=(1180/dpi,480/dpi), dpi=dpi)
        plt.rcParams.update({'font.size': 16})
        plt.rcParams["axes.linewidth"] = 1
    
        ax = plt.gca()
    
        ax.scatter(trimmedPSD['f [Hz]'],unumpy.nominal_values(trimmedPSD['power [V**2/Hz]']),label = 'trimmed PSD' , s = 10)
        ax.plot(trimmedPSD['f [Hz]'],modelSimplified(trimmedPSD['f [Hz]'],ans[0],ans[1],ans[2],ans[3]), 'r',label='fitted function')
        ax.set_ylim([min(unumpy.nominal_values(trimmedPSD['power [V**2/Hz]'])), 2*max(unumpy.nominal_values(trimmedPSD['power [V**2/Hz]']))])
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()
        ax.set(ylabel=r'$V^2$/Hz')
        ax.set(xlabel=r'$\Omega/2\pi$ [Hz]')
        ax.grid(alpha = 0.4)
        plt.tight_layout()
    
        outputFile = os.path.join(outputFolder,'trimmedPSDwithFIT.png')
        plt.savefig(outputFile)
        plt.close()
        
        #placing plot
        mainWindow.resultPlotDetCal.setPixmap(QtGui.QPixmap(outputFile))
        mainWindow.stackedWidget.setCurrentIndex(2)  
        mainWindow.progressStatus_B_2.setText("Detector calibration completed! Please refer to the results tab!")
        mainWindow.label_31.setText('\nThe calibration factor is: {:.2u}'.format(calibrationFactor*1e-6)+ " [mV/nm]. For more details, press the button below.")
        
        mainWindow.label_34.setText(str(unumpy.nominal_values(calibrationFactor)))
        mainWindow.label_35.setText(str(unumpy.std_devs(calibrationFactor)))
        
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
                 'The calibration factor is: {:.2u}'.format(calibrationFactor*1e-6)+ " [mV/nm]",
                 'Hint D is '+str(hint[0]),
                 'Hint gamma is '+str(hint[1]),
                 'Hint f_0 is '+str(hint[2]),
                 'Hint cst is '+str(hint[3]),
                 ]

        with open(os.path.join(outputFolder,'experimentInfo.txt'), 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
        
    except NameError:
        QMessageBox.warning(mainWindow,'Warning','Some error with the fitting occured, please try again.')
                    
app = QtWidgets.QApplication([])

#initialize windows
mainWindow = uic.loadUi("tiepieUI.ui")
helpWindow = uic.loadUi("helpWindow.ui")
cutFrequencyWindow = uic.loadUi("plotWindow_cut.ui")

#connect buttons
mainWindow.startAcqButton_A.clicked.connect(startAcquisition)
mainWindow.browseFolderButton_A.clicked.connect(getFolderName)
mainWindow.browseFolderButton_B_2.clicked.connect(getFolderName)
mainWindow.actionhelp.triggered.connect(openHelpWindow)
mainWindow.startDetCal_B.clicked.connect(startDetCal)
mainWindow.LRbutton.clicked.connect(getLeftRightCut)

mainWindow.pushButton.clicked.connect(parametersWindow_detCal)
mainWindow.pushButton_2.clicked.connect(leftRightWindow_detCal)
mainWindow.pushButton_3.clicked.connect(resultsWindow_detCal)
mainWindow.pushButton_4.clicked.connect(parametersWindow_detCal)
mainWindow.pushButton_5.clicked.connect(leftRightWindow_detCal)
mainWindow.pushButton_6.clicked.connect(resultsWindow_detCal)
mainWindow.pushButton_7.clicked.connect(results2Window_detCal)
mainWindow.pushButton_9.clicked.connect(resultsWindow_detCal)


#initialize main window
mainWindow.show()
#sys.exit(app.exec())