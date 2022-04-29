import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

d = 'Apr-28-2022' #data que a coleta foi feita
experiment = '7' #número do experimento desejado


#names of the root folder, main folder and file where the traces are saved
root_folder = 'C:/Users/Labq/Desktop/Felipe/python scripts' #PC do laboratório
#root_folder = '/Users/tandeitnik/Dropbox/Daniel_RT.T/scripts/new_scripts' #MacBook Air pessoal
folder_name = 'exp_'+experiment+'_'+d
#file_name = 'traces_CH1-CH2-CH3.npz'
file_name = 'traces_CH2-CH3-CH4.npz'

#loading data
data = np.load(root_folder+'/'+folder_name+'/'+file_name)

#extracting compressed data variables

traces_CH2 = data['arr_0']
traces_CH3 = data['arr_1']
traces_CH4 = data['arr_2']

time = traces_CH2[0][0] #array with time stamps
signal_x_norm = traces_CH2[:,1,:]/traces_CH4[:,1,:] #normalized signal x
signal_y_norm = traces_CH3[:,1,:]/traces_CH4[:,1,:] #normalized signal y
#signal_x_norm = traces_CH2[:,1,:]
#signal_y_norm = traces_CH3[:,1,:]

#defining PSD function. It takes a NxM array (N traces x M time stamps) and returns the frequencies and a list with M entries, each entry has the PSD of each trace. It's assumed that all signals have the same time stamps. 
def psd(time, sig):
    dt = time[1]-time[0]
    p_den = []
    for i in range(len(sig)):
        freq, power = signal.welch(sig[i,:], 1/dt, window = 'rectangular', nperseg = len(sig[i]))
        p_den.append(power)
    return freq, p_den

#defining mean function. It takes a list of arrays of same size and evaluates the mean of the arrays entry per entry
def mean_list(lista):
    array_soma = lista[0]
    for i in range(len(lista)-1):
        array_soma += lista[i+1]
    media = array_soma/len(lista)
    return media

#calculate psds
freq_x, psds_x = psd(time, signal_x_norm)
freq_y, psds_y = psd(time, signal_y_norm)

#calculate the mean
mean_psd_x = mean_list(psds_x)
mean_psd_y = mean_list(psds_y)

#fit lorentz
def model(f,D,f_c,cst):
    return  (D/(2*np.pi**2))/(f_c**2+f**2)+ cst

fit_x = curve_fit(model,freq_x,mean_psd_x)
fit_y = curve_fit(model,freq_y,mean_psd_y)

ans_x, cov_x = fit_x
ans_y, cov_y = fit_y
fit_D_x,fit_f_c_x,fit_cst_x = ans_x
fit_D_y,fit_f_c_y,fit_cst_y = ans_y

print("F_c_x = "+str(fit_f_c_x))
print("F_c_y = "+str(fit_f_c_y))

psd_fit_x = np.zeros(len(freq_x))
psd_fit_y = np.zeros(len(freq_y))

for i in range(len(freq_x)):
    psd_fit_x[i] = model(freq_x[i],fit_D_x,fit_f_c_x,fit_cst_x)
for i in range(len(freq_y)):
    psd_fit_y[i] = model(freq_y[i],fit_D_y,fit_f_c_y,fit_cst_y)

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

ax3.loglog(freq_x[1:], mean_psd_x[1:])
ax3.loglog(freq_x[1:],psd_fit_x[1:])
ax3.set_title("PSD - x direction")
fig3.savefig(root_folder+'/'+folder_name+'/'+'psd_x_loglog.png')

ax4.loglog(freq_y[1:], mean_psd_y[1:])
ax4.loglog(freq_y[1:],psd_fit_y[1:])
ax4.set_title("PSD - y direction")
fig4.savefig(root_folder+'/'+folder_name+'/'+'psd_y_loglog.png')
