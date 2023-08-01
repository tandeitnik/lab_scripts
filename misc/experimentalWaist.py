# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:31:57 2023

@author: tandeitnik

Description: This script evaluates the waist of a Gaussian beam using photos taken from it at a single transversal position. It uses the photos to evaluate
a mean image with smoother contrast.
"""

import numpy as np
import os
import imageio.v3 as iio
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm

rootFolder = r"D:\gaussianPhotos" #folder with photos of the beam
boxLengthX = 800 #horizontal length of the cropping box
boxLengthY = 800 #vertical length of the cropping box
pixelPitch = 3.75e-6



root, dump, files = next(os.walk(rootFolder))
meanImage = np.zeros([boxLengthY,boxLengthX])
skip = 0

for i in tqdm(range(len(files))):
    
    file = os.path.join(root,files[i]) #getting name of the file
    im = np.array(iio.imread(file)) #importing file
    
    #getting centroid
    r, c = im.shape
    x = np.linspace(0,c-1,c)
    y = np.linspace(0,r-1,r)
    X,Y = np.meshgrid(x,y)
    centX = int(np.sum(X*im)/np.sum(im)) #central column
    centY = int(np.sum(Y*im)/np.sum(im)) #central row
    
    #cropping and summing to the mean
    try:
        centeredImage = im[int(centY-boxLengthY/2):int(centY+boxLengthY/2), int(centX-boxLengthX/2):int(centX+boxLengthX/2)]
        meanImage += centeredImage
    except:
        skip += 1
    
meanImage = meanImage/(len(files)-skip) #final result
meanImage = meanImage/np.max(meanImage) #normalizing

from scipy.optimize import curve_fit
def gauss(x, cst, A, x0, W):
    return cst + A * np.exp(-2*(x - x0)**2 / W**2)

parameters, covariance = curve_fit(gauss, np.linspace(0,boxLengthX,boxLengthX), meanImage[centY,:], p0 = [0,1,400,50])
fit = gauss(np.linspace(0,boxLengthX,boxLengthX),parameters[0],parameters[1],parameters[2],parameters[3])


evenly_spaced_interval = np.linspace(0, 1, 10)
colors = [cm.viridis(x) for x in evenly_spaced_interval]

fig, ax = plt.subplots(1,1, figsize=(7,4), sharex=False)
    
plt.rcParams.update({'font.size': 20})
plt.rcParams["axes.linewidth"] = 1

ax.scatter(np.linspace(0,boxLengthX,boxLengthX)*pixelPitch*1e3, meanImage[centY,:],label = 'experimental data')
ax.plot(np.linspace(0,boxLengthX,boxLengthX)*pixelPitch*1e3, fit,label = 'Gaussian fit', color = 'red')
ax.set(xlabel='x [mm]')
ax.set(ylabel='normalized power')
ax.grid(alpha = 0.4)
ax.legend(loc = 'upper right')
ax.set(title = "")

fig.tight_layout()
fig.subplots_adjust(hspace=0.1)

waist = parameters[-1]*pixelPitch
waistError = np.sqrt(covariance[-1,-1])*pixelPitch
