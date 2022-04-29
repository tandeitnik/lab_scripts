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

sample_frequency = 50e2 #quantidade de pontos por traço
acq_time = 0.5 # tempo total do traço

path = r"C:\Users\Labq\Desktop\Felipe\python scripts\TiePie"  #pasta raíz aonde a nova pasta será criada


freq = sample_frequency/acq_time
dt = 1/freq

skip_line = 2
files = next(os.walk(path))[2]
data = []

for file in files:

    data.append(np.genfromtxt(path+'/'+file, delimiter=';', skip_header=skip_line))
    
freq, power = signal.welch(data[0][:,1], 1/dt, window = 'rectangular', nperseg = len(data[0][:,1]))

PSD_x = np.zeros([len(data),len(freq)-1])
PSD_y = np.zeros([len(data),len(freq)-1])

for i in range(len(data)):
    
    PSD_x[i,:] = signal.welch(data[i][:,1], 1/dt, window = 'rectangular', nperseg = len(data[0][:,1]))[1][1:]
    PSD_y[i,:] = signal.welch(data[i][:,2], 1/dt, window = 'rectangular', nperseg = len(data[0][:,1]))[1][1:]


psd_mean_x = PSD_x[0,:]
psd_mean_y = PSD_y[0,:]
    
for i in range(1,len(data)):
    
    psd_mean_x += PSD_x[i,:]
    psd_mean_y += PSD_y[i,:]
    
psd_mean_x = psd_mean_x/len(data)
psd_mean_y = psd_mean_y/len(data)



def model(f,D,f_c,cst):
    return  (D/(2*np.pi**2))/(f_c**2+f**2)+ cst

fit_x = curve_fit(model,freq[1:],psd_mean_x)
fit_y = curve_fit(model,freq[1:],psd_mean_y)

ans_x, cov_x = fit_x
ans_y, cov_y = fit_y
fit_D_x,fit_f_c_x,fit_cst_x = ans_x
fit_D_y,fit_f_c_y,fit_cst_y = ans_y

print("F_c_x = "+str(fit_f_c_x)+" +- "+str(np.sqrt(cov_x[1,1])))
print("F_c_y = "+str(fit_f_c_y)+" +- "+str(np.sqrt(cov_x[1,1])))

psd_fit_x = np.zeros(len(freq[1:]))
psd_fit_y = np.zeros(len(freq[1:]))

for i in range(1,len(freq)-1):
    psd_fit_x[i] = model(freq[i],fit_D_x,fit_f_c_x,fit_cst_x)
for i in range(len(freq)-1):
    psd_fit_y[i] = model(freq[i],fit_D_y,fit_f_c_y,fit_cst_y)

#plot results and save plots
#fig1, ax1 = plt.subplots()
#fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

#ax1.plot(freq_x, mean_psd_x)
#ax1.set_title("PSD - x direction")
#fig1.savefig(root_folder+'/'+folder_name+'/'+'psd_x.png')

#ax2.plot(freq_y, mean_psd_y)
#ax2.set_title("PSD - y direction")
#fig2.savefig(root_folder+'/'+folder_name+'/'+'psd_y.png')

"""
Os plots abaixo desconsideram o primeiro ponto, pois ele retorna com um valor
extremamente próximo de 0, o que atrapalha na escala no gráfico.
"""

ax3.loglog(freq[1:], psd_mean_x)
ax3.loglog(freq[1:],psd_fit_x)
ax3.set_title("PSD - x direction")
fig3.savefig(root_folder+'/'+folder_name+'/'+'psd_x_loglog.png')

ax4.loglog(freq[1:], psd_mean_y)
ax4.loglog(freq[1:],psd_fit_y)
ax4.set_title("PSD - y direction")
fig4.savefig(root_folder+'/'+folder_name+'/'+'psd_y_loglog.png')
