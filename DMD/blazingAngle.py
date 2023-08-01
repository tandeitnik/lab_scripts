# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:25:24 2023

@author: tandeitnik

Description: This script calculates the incident/reflected for the blazing condition of the DMD
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#parameters
alpha = np.radians(12) #blazing angle of the surface
m = 1 #diffraction order to maximize
wl = 1550e-9 #wavelength of incident light
d = 5.4e-6 #grating spacing


theta_i = np.linspace(0,np.pi/2,10000) #possible incident angles
thetaSpecular = np.degrees(2*alpha - theta_i) #specular reflection
thetaGrating = np.degrees(np.arcsin( m*wl/d - np.sin(theta_i))) #diffraction reflection

#determining crossing point (for which incident angle the output angle obeys specular reflection)
crossoverTest = np.pad(np.diff(np.array(thetaSpecular > thetaGrating).astype(int)), (1,0), 'constant', constant_values = (0,))
idx = [i for i in range(len(crossoverTest)) if crossoverTest[i] != 0]

#plotting
evenly_spaced_interval = np.linspace(0, 1, 10)
colors = [cm.viridis(x) for x in evenly_spaced_interval]

fig, ax = plt.subplots(1,1, figsize=(7,4), sharex=False)
    
plt.rcParams.update({'font.size': 20})
plt.rcParams["axes.linewidth"] = 1

ax.plot(np.degrees(theta_i),thetaSpecular,color = colors[0], alpha = 1,lw = 2, label = 'specular reflection')
ax.plot(np.degrees(theta_i), thetaGrating,color = colors[-1], alpha = 1,lw = 2, label = 'grating reflection')
for i in range(len(idx)): #even though I used a loop, there should be only one solution
    
    ax.scatter(np.degrees(theta_i[idx[i]]),thetaSpecular[idx[i]], color = 'black', label = 'blazing angle')

ax.set(xlabel=r'$\theta_{in}$ [deg.]')
ax.set(ylabel=r'$\theta_{out}$ [deg.]')
ax.grid(alpha = 0.4)
ax.legend(loc = 'upper right')
ax.set(title = "")

fig.tight_layout()
fig.subplots_adjust(hspace=0.1)

#the solution
try:
    blazingThetaIncident = np.degrees(theta_i[idx[0]])
    blazingThetaOutput = thetaSpecular[idx[0]]
    print('for m='+str(m)+' inc = '+str(blazingThetaIncident)+' out = '+str(blazingThetaOutput))
except:
    print("there is no solution for m = "+str(m))
    
