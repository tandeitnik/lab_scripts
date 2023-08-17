# -*- coding: utf-8 -*-
"""
Created on Fri May  5 06:45:05 2023

@author: tandeitnik
"""

import numpy as np
from scipy.special import hermite
from math import factorial
from scipy.special import eval_genlaguerre
import matplotlib.pyplot as plt
import matplotlib.image
from PIL import Image
import os

def scaleConv(image,bits):
    
    hValue = 2**bits - 1
    
    return (hValue/(np.max(image)-np.min(image)))*(image - np.min(image))

def floydDithering(image, bits, s = 0):
    
    #if s = 1 it alternates de direction of the lines
    
    hValue = 2**bits - 1
    threshhold = hValue/2
    
    y,x = np.shape(image)
    
    if s == 0:
    
        for i in range(y):
            
            for j in range(x):
                
                oldpixel = image[i,j]
                if oldpixel <= threshhold:
                    newpixel = 0
                else:
                    newpixel = 1

                image[i,j] = newpixel
                quant_error = oldpixel - newpixel*hValue
                if j != x-1:
                    image[i,j+1] = image[i,j+1] + quant_error*(7/16)
                if (i != y-1) and (j != 0):
                    image[i+1,j-1] = image[i+1,j-1] + quant_error*(3/16)
                if i != y-1:
                    image[i+1,j] = image[i+1,j] + quant_error*(5/16)
                if (i != y-1) and (j != x-1):
                    image[i+1,j+1] = image[i+1,j+1] + quant_error*(1/16)
                
    if s == 1:
        
        for i in range(y):
            
            for j in range(x):
                
                if i % 2 == 0:
                
                    oldpixel = image[i,j]
                    if oldpixel <= threshhold:
                        newpixel = 0
                    else:
                        newpixel = 1
                        
                    image[i,j] = newpixel
                    quant_error = oldpixel - newpixel*hValue
                    if j != x-1:
                        image[i,j+1] = image[i,j+1] + quant_error*(7/16)
                    if (i != y-1) and (j != 0):
                        image[i+1,j-1] = image[i+1,j-1] + quant_error*(3/16)
                    if i != y-1:
                        image[i+1,j] = image[i+1,j] + quant_error*(5/16)
                    if (i != y-1) and (j != x-1):
                        image[i+1,j+1] = image[i+1,j+1] + quant_error*(1/16)
                        
                else:
                    
                    oldpixel = image[i,-1 - j]
                    if oldpixel <= threshhold:
                        newpixel = 0
                    else:
                        newpixel = 1
                        
                    image[i,-1 - j] = newpixel
                    quant_error = oldpixel - newpixel*hValue
                    if j != x-1:
                        image[i,-1 - j-1] = image[i,-1 - j-1] + quant_error*(7/16)
                    if (i != y-1) and (j != 0):
                        image[i+1,-1 - j+1] = image[i+1,-1 - j+1] + quant_error*(3/16)
                    if i != y-1:
                        image[i+1,-1 - j] = image[i+1,-1 - j] + quant_error*(5/16)
                    if (i != y-1) and (j != x-1):
                        image[i+1,-1 - j-1] = image[i+1,-1 - j-1] + quant_error*(1/16)
        
    return image
    
def jarvisDithering(image, bits, s = 0):
    
    #if s = 1 it alternates de direction of the lines
    
    hValue = 2**bits - 1
    threshhold = hValue/2
    
    y,x = np.shape(image)
    
    if s == 0:
    
        for i in range(y):
            
            for j in range(x):
                
                oldpixel = image[i,j]
                if oldpixel <= threshhold:
                    newpixel = 0
                else:
                    newpixel = 1

                image[i,j] = newpixel
                quant_error = oldpixel - newpixel*hValue
                if j != x-1:
                    image[i,j+1] = image[i,j+1] + quant_error*(7/48)
                if j < x-2:
                    image[i,j+2] = image[i,j+2] + quant_error*(5/48)
                
                if (i < y-1) and (j >= 2):
                    image[i+1,j-2] = image[i+1,j-2] + quant_error*(3/48)
                if (i < y-1) and (j >= 1):
                    image[i+1,j-1] = image[i+1,j-1] + quant_error*(5/48)
                if (i < y-1):
                    image[i+1,j] = image[i+1,j] + quant_error*(7/48)
                if i < y-1  and (j < x-1):
                    image[i+1,j+1] = image[i+1,j+1] + quant_error*(5/48)
                if (i < y-1) and (j < x-2):
                    image[i+1,j+2] = image[i+1,j+2] + quant_error*(3/48)
                    
                if (i < y-2) and (j >= 2):
                    image[i+2,j-2] = image[i+2,j-2] + quant_error*(1/48)
                if (i < y-2) and (j >= 1):
                    image[i+2,j-1] = image[i+2,j-1] + quant_error*(3/48)
                if (i < y-2):
                    image[i+2,j] = image[i+2,j] + quant_error*(5/48)
                if (i < y-2)  and (j < x-1):
                    image[i+2,j+1] = image[i+2,j+1] + quant_error*(3/48)
                if (i < y-2) and (j < x-2):
                    image[i+2,j+2] = image[i+2,j+2] + quant_error*(1/48)
                
    if s == 1:
        
        for i in range(y):
            
            for j in range(x):
                
                if i % 2 == 0:
                
                    oldpixel = image[i,j]
                    if oldpixel <= threshhold:
                        newpixel = 0
                    else:
                        newpixel = 1
                        
                    image[i,j] = newpixel
                    quant_error = oldpixel - newpixel*hValue
                    if j != x-1:
                        image[i,j+1] = image[i,j+1] + quant_error*(7/48)
                    if j < x-2:
                        image[i,j+2] = image[i,j+2] + quant_error*(5/48)
                    
                    if (i < y-1) and (j >= 2):
                        image[i+1,j-2] = image[i+1,j-2] + quant_error*(3/48)
                    if (i < y-1) and (j >= 1):
                        image[i+1,j-1] = image[i+1,j-1] + quant_error*(5/48)
                    if (i < y-1):
                        image[i+1,j] = image[i+1,j] + quant_error*(7/48)
                    if i < y-1  and (j < x-1):
                        image[i+1,j+1] = image[i+1,j+1] + quant_error*(5/48)
                    if (i < y-1) and (j < x-2):
                        image[i+1,j+2] = image[i+1,j+2] + quant_error*(3/48)
                        
                    if (i < y-2) and (j >= 2):
                        image[i+2,j-2] = image[i+2,j-2] + quant_error*(1/48)
                    if (i < y-2) and (j >= 1):
                        image[i+2,j-1] = image[i+2,j-1] + quant_error*(3/48)
                    if (i < y-2):
                        image[i+2,j] = image[i+2,j] + quant_error*(5/48)
                    if (i < y-2)  and (j < x-1):
                        image[i+2,j+1] = image[i+2,j+1] + quant_error*(3/48)
                    if (i < y-2) and (j < x-2):
                        image[i+2,j+2] = image[i+2,j+2] + quant_error*(1/48)
                        
                else:
                    
                    
                    
                    oldpixel = image[i,-1 - j]
                    if oldpixel <= threshhold:
                        newpixel = 0
                    else:
                        newpixel = 1
                    
                    image[i,-1 - j] = newpixel
                    quant_error = oldpixel - newpixel*hValue
                    
                    if j != x-1:
                        image[i,-1 - j-1] = image[i,-1 - j-1] + quant_error*(7/48)
                    if j < x-2:
                        image[i,-1 - j-2] = image[i,-1 - j-2] + quant_error*(5/48)
                    
                    if (i < y-1) and (j >= 2):
                        image[i+1,-1 - j+2] = image[i+1,-1 - j+2] + quant_error*(3/48)
                    if (i < y-1) and (j >= 1):
                        image[i+1,-1 - j+1] = image[i+1,-1 - j+1] + quant_error*(5/48)
                    if (i < y-1):
                        image[i+1,-1 - j] = image[i+1,-1 - j] + quant_error*(7/48)
                    if i < y-1  and (j < x-1):
                        image[i+1,-1 - j-1] = image[i+1,-1 - j-1] + quant_error*(5/48)
                    if (i < y-1) and (j < x-2):
                        image[i+1,-1 - j-2] = image[i+1,-1 - j-2] + quant_error*(3/48)
                        
                    if (i < y-2) and (j >= 2):
                        image[i+2,-1 - j+2] = image[i+2,-1 - j+2] + quant_error*(1/48)
                    if (i < y-2) and (j >= 1):
                        image[i+2,-1 - j+1] = image[i+2,-1 - j+1] + quant_error*(3/48)
                    if (i < y-2):
                        image[i+2,-1 - j] = image[i+2,-1 - j] + quant_error*(5/48)
                    if (i < y-2)  and (j < x-1):
                        image[i+2,-1 - j-1] = image[i+2,-1 - j-1] + quant_error*(3/48)
                    if (i < y-2) and (j < x-2):
                        image[i+2,-1 - j-2] = image[i+2,-1 - j-2] + quant_error*(1/48)
                    
    return image

def amplitudeGaussianHermite(x,y,z,wl,w0,h = 0, v = 0, incidentAngle = 0):

    n = 0
    m = 0
    correctionFactor = 1/np.cos(np.radians(incidentAngle))
    X, Y = np.meshgrid((x-h)/correctionFactor, y-v)
    
    k = (2*np.pi)/wl
    z_r = (k*w0**2)/2
    w = w0*np.sqrt(1 + z**2/z_r**2)
    H_n = hermite(n, monic = True)
    H_m = hermite(m, monic = True)
    
        
    amplitude = (w0/w)*np.exp(-(X**2+Y**2)/w**2)*H_m(np.sqrt(2)*X/w)*H_n(np.sqrt(2)*Y/w)
    
    return amplitude

def amplitudeGaussianLaguerre(x,y,z,wl,w0,h = 0, v = 0, incidentAngle = 0):

    p = 0
    l = 0
    correctionFactor = 1/np.cos(np.radians(incidentAngle))
    X, Y = np.meshgrid((x-h)/correctionFactor, y-v)
    
    r = np.sqrt(X**2+Y**2)
    
    C = np.sqrt((2*factorial(p))/(np.pi*factorial(p+abs(l)))) #normalization constant
    k = (2*np.pi)/wl
    z_r = (k*w0**2)/2
    w = w0*np.sqrt(1 + z**2/z_r**2)
    laguerre = eval_genlaguerre(p, abs(l), 2*r**2/w**2)
    
    amplitude = (C/w)*np.exp(((-r**2)/w**2))*((np.sqrt(2)*r/w)**abs(l))*laguerre
    
    return amplitude

def ditHermiteHologram(n,m,wl,w0,z,x,y,x0,y0,theta = 0,h = 0,v = 0,waistGaussian = 0,incidentAngle = 0,invert = 0,save = 0, saveFileName = 'ditHermiteHologram'):

    Xtemp, Ytemp = np.meshgrid(x, y)
    X = (Xtemp*np.cos(np.radians(theta)) - Ytemp*np.sin(np.radians(theta)))
    Y = (Xtemp*np.sin(np.radians(theta)) + Ytemp*np.cos(np.radians(theta)))
    
    k = (2*np.pi)/wl
    z_r = (k*w0**2)/2
    w = w0*np.sqrt(1 + z**2/z_r**2)
    H_n = hermite(n, monic = True)
    H_m = hermite(m, monic = True)
    
    if z != 0:
        R = (z_r**2 + z**2)/z
        
    amplitude = (w0/w)*np.exp(-(X**2+Y**2)/w**2)*H_m(np.sqrt(2)*X/w)*H_n(np.sqrt(2)*Y/w)
    ampAbsNorm = abs(amplitude)/np.max(abs(amplitude))
    
    if z != 0:
        phase = (-k*(X**2+Y**2))/(2*R) + (1+n+m)*np.arctan2(z,z_r) - k*z - (np.sign(np.sign(amplitude)+0.1)-1)*(np.pi/2)
    else:
        phase =  (1+n+m)*np.arctan2(z,z_r) - k*z -  (np.sign(np.sign(amplitude)+0.1)-1)*(np.pi/2)
    
    phasePlane = 2*np.pi*Xtemp/x0 + 2*np.pi*Ytemp/y0
    
    if waistGaussian != 0:
        
        amplitudeGaussian = amplitudeGaussianHermite(x,y,z,wl,w0 = waistGaussian,h = h,v = v, incidentAngle = incidentAngle)
        amplitudeGaussian = amplitudeGaussian/np.max(amplitudeGaussian)
        ampAbsNorm = ampAbsNorm/amplitudeGaussian
        ampAbsNorm = ampAbsNorm/np.max(ampAbsNorm)
        
    hologram = 1 + ampAbsNorm**2 + 2*ampAbsNorm*np.cos(phasePlane-phase)
    
    hologram = scaleConv(hologram,8) #converting to 8bit
    hologram = jarvisDithering(hologram,8, s= 1) #dithering
    
    if invert == 1:
        
        hologram = (hologram+1) % 2
    
    if save == 1:
        plt.gray()
        matplotlib.image.imsave(os.path.join(os.getcwd(),"tempFile.png"), hologram)
        plt.close()
        img = Image.open(os.path.join(os.getcwd(),"tempFile.png")).convert("RGB")
        img.save(os.path.join(os.getcwd(),saveFileName+".bmp"))
        os.remove(os.path.join(os.getcwd(),"tempFile.png"))
    
    return hologram

def ditLaguerreHologram(l,p,wl,w0,z,x,y,x0,y0,h = 0,v = 0,waistGaussian = 0,incidentAngle = 0, invert = 0, save = 0,saveFileName = 'ditLaguerreHologram'):

    X, Y = np.meshgrid(x, y)
    
    r = np.sqrt(X**2+Y**2)
    phi = np.arctan2(Y,X)
    
    C = np.sqrt((2*factorial(p))/(np.pi*factorial(p+abs(l)))) #normalization constant
    k = (2*np.pi)/wl
    z_r = (k*w0**2)/2
    w = w0*np.sqrt(1 + z**2/z_r**2)
    laguerre = eval_genlaguerre(p, abs(l), 2*r**2/w**2)
    
    if z != 0:
        R = (z_r**2 + z**2)/z
        
    Phi = (2*p+abs(l)+1)*np.arctan2(z,z_r)
    
    amplitude = (C/w)*np.exp(((-r**2)/w**2))*((np.sqrt(2)*r/w)**abs(l))*laguerre
    ampAbsNorm = abs(amplitude)/np.max(abs(amplitude))
    
    if z != 0:
        phase = (-k*r**2)/(2*R) + Phi + l*phi - k*z -  (np.sign(np.sign(amplitude)+0.1)-1)*(np.pi/2)
    else:
        phase =  (Phi + l*phi) - k*z - (np.sign(np.sign(amplitude)+0.1)-1)*(np.pi/2)
    
    phasePlane = 2*np.pi*X/x0 + 2*np.pi*Y/y0
    
    if waistGaussian != 0:
        
        amplitudeGaussian = amplitudeGaussianLaguerre(x,y,z,wl,w0 = waistGaussian,h = h,v = v, incidentAngle = incidentAngle)
        amplitudeGaussian = amplitudeGaussian/np.max(amplitudeGaussian)
        ampAbsNorm = ampAbsNorm/amplitudeGaussian
        ampAbsNorm = ampAbsNorm/np.max(ampAbsNorm)
        
    hologram = 1 + ampAbsNorm**2 + 2*ampAbsNorm*np.cos(phasePlane-phase)
    
    hologram = scaleConv(hologram,8) #converting to 8bit
    hologram = jarvisDithering(hologram,8, s= 1) #dithering
    
    if invert == 1:
        
        hologram = (hologram+1) % 2
    
    if save == 1:
        plt.gray()
        matplotlib.image.imsave(os.path.join(os.getcwd(),"tempFile.png"), hologram)
        plt.close()
        img = Image.open(os.path.join(os.getcwd(),"tempFile.png")).convert("RGB")
        img.save(os.path.join(os.getcwd(),saveFileName+".bmp"))
        os.remove(os.path.join(os.getcwd(),"tempFile.png"))
    
    return hologram

def grating(wl,x,y,x0,y0,centralCircle = 0, radius = 0,invert = 0,save = 0,saveFileName = 'grating'):
    
    X, Y = np.meshgrid(x, y)
    
    phasePlane = 2*np.pi*X/x0 + 2*np.pi*Y/y0
    
    hologram = 2 + 2*np.cos(phasePlane)
    
    hologram = scaleConv(hologram,8) #converting to 8bit
    hologram = jarvisDithering(hologram,8, s= 1) #dithering
    
    if centralCircle == 1:
    
        for i in range(len(y)):
            
            for j in range(len(x)):
                
                if np.sqrt((y[i])**2+(x[j])**2) >= radius:
                    
                    hologram[i,j] = 0
                    
    if invert == 1:
        
        hologram = (hologram+1) % 2
    
    if save == 1:
        plt.gray()
        matplotlib.image.imsave(os.path.join(os.getcwd(),"tempFile.png"), hologram)
        plt.close()
        img = Image.open(os.path.join(os.getcwd(),"tempFile.png")).convert("RGB")
        img.save(os.path.join(os.getcwd(), saveFileName+".bmp"))
        os.remove(os.path.join(os.getcwd(),"tempFile.png"))
    
    return hologram
