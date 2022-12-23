# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:35:24 2022

@author: Labq
"""
import os
import numpy as np
from scipy.optimize import curve_fit
import scipy.signal as signal
import matplotlib.pyplot as plt

#ample_frequency = 50e2 #quantidade de pontos por traço
#cq_time = 0.5 # tempo total do traço

path = r"C:\Users\Labq\Dropbox\scripts\tiepie\data"  #pasta raíz aonde contém os dados
N = 1 #quantos traços fazer por traço
normalize = 1 #se 0 não normaliza o sinal do canal 1 com o canal 2, se 1 normaliza
skip_points = 7 # número de pontos iniciais ignorados pro fit




skip_line = 2
files = next(os.walk(path))[2]
data = []

PSD_list = []

for file in files:

    data = np.genfromtxt(path+'/'+file, delimiter=';', skip_header=skip_line)
    
    dt = data[1,1]
    rowsPerTrace = int(len(data)/N)
    
    #calculando o PSD assumindo que os dados foram pegos numa QPD, que o sinalesteja no canal 1 e que a soma no canal 2
    for i in range(N):
        
        if normalize == 1:
            freq, power = signal.welch(data[i*rowsPerTrace:(i+1)*rowsPerTrace,2]/data[i*rowsPerTrace:(i+1)*rowsPerTrace,3], 1/dt, window = 'rectangular', nperseg = rowsPerTrace)
        else:    
            freq, power = signal.welch(data[i*rowsPerTrace:(i+1)*rowsPerTrace,2], 1/dt, window = 'rectangular', nperseg = rowsPerTrace)
        if i == 0:
            
            PSD  = power
            
        else:
            
            PSD  += power
    
    #tirando a média
    PSD = PSD/N
    
    PSD_list.append(PSD[skip_points:])
    
mean_psd = PSD_list[0]

for i in range(1,len(files)):
    
    mean_psd = mean_psd + PSD_list[i]
    
mean_psd = mean_psd/len(files)

var_psd = (PSD_list[0]-mean_psd)**2

for i in range(1,len(files)):
    
    var_psd = var_psd + (PSD_list[i]-mean_psd)**2
    
var_psd = var_psd/len(files)
      
def aliased_lorentzian(f, D, f_c, cst ):                                       #Gives the function that replaces the Lorentzian in the case of finite sampling frequency, and it should fit the experimental spectrum for all frequencies of 0,f k<f Nyq if            # D is not a trustable value because it carries information about the conversion factor between position and milivolts units
    global dt                                                                  #Sampling interval   
    f_s=1/dt                                                                   #Sampling frequency     
    c=np.exp(-2*np.pi*f_c/f_s)                                                 #Exponential that carries information about f_c
    k=f*dt                                                                     #The frequencies in which PSD is evaluated multiplied by dt  
    return (1-c**2)*D/(2*np.pi*f_c*f_s)*1/(1+c**2-2*c*np.cos(2*np.pi*k))+cst

fit = curve_fit(aliased_lorentzian,freq[skip_points:],mean_psd)
ans, cov = fit
   
D,f_c,cst = ans    
perr = np.sqrt(np.diag(cov))        

fig, ax = plt.subplots()
ax.errorbar(freq[skip_points:], mean_psd, fmt='o', markersize=1, color='red', ecolor='red', elinewidth=0.5, capsize=0)
ax.plot(freq[skip_points:],aliased_lorentzian(freq[skip_points:], D, f_c, cst ), color='blue')
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
