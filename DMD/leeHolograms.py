# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 13:50:28 2023

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

def amplitudeGaussianHermite(x,y,z,wl,w0,horzScaleFactor = 1,vertScaleFactor = 1):
    """
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    wl : TYPE
        DESCRIPTION.
    w0 : TYPE
        DESCRIPTION.
    horzScaleFactor : TYPE, optional
        DESCRIPTION. The default is 1.
    vertScaleFactor : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    amplitude : TYPE
        DESCRIPTION.

    """
    
    
    n = 0
    m = 0
        
    X, Y = np.meshgrid(x/horzScaleFactor, y/vertScaleFactor)
    
    k = (2*np.pi)/wl
    z_r = (k*w0**2)/2
    w = w0*np.sqrt(1 + z**2/z_r**2)
    H_n = hermite(n, monic = True)
    H_m = hermite(m, monic = True)
    
        
    amplitude = (w0/w)*np.exp(-(X**2+Y**2)/w**2)*H_m(np.sqrt(2)*X/w)*H_n(np.sqrt(2)*Y/w)
    
    return amplitude

def amplitudeGaussianLaguerre(x,y,z,wl,w0,horzScaleFactor = 1,vertScaleFactor = 1):
    """
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    wl : TYPE
        DESCRIPTION.
    w0 : TYPE
        DESCRIPTION.
    horzScaleFactor : TYPE, optional
        DESCRIPTION. The default is 1.
    vertScaleFactor : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    amplitude : TYPE
        DESCRIPTION.

    """
    p = 0
    l = 0
    
    X, Y = np.meshgrid(x/horzScaleFactor, y/vertScaleFactor)
    
    r = np.sqrt(X**2+Y**2)
    
    C = np.sqrt((2*factorial(p))/(np.pi*factorial(p+abs(l)))) #normalization constant
    k = (2*np.pi)/wl
    z_r = (k*w0**2)/2
    w = w0*np.sqrt(1 + z**2/z_r**2)
    laguerre = eval_genlaguerre(p, abs(l), 2*r**2/w**2)
    
    amplitude = (C/w)*np.exp(((-r**2)/w**2))*((np.sqrt(2)*r/w)**abs(l))*laguerre
    
    return amplitude

def grating(wl,x,y,x0,y0,centralCircle = 0, radius = 0,horzScaleFactor = 1,verticalFactor = 1,save = 0,saveFileName = 'grating'):
    """
    Parameters
    ----------
    wl : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    x0 : TYPE
        DESCRIPTION.
    y0 : TYPE
        DESCRIPTION.
    centralCircle : TYPE, optional
        DESCRIPTION. The default is 0.
    radius : TYPE, optional
        DESCRIPTION. The default is 0.
    horzScaleFactor : TYPE, optional
        DESCRIPTION. The default is 1.
    verticalFactor : TYPE, optional
        DESCRIPTION. The default is 1.
    save : TYPE, optional
        DESCRIPTION. The default is 0.
    saveFileName : TYPE, optional
        DESCRIPTION. The default is 'grating'.

    Returns
    -------
    hologram : TYPE
        DESCRIPTION.

    """
    amplitude = np.ones([len(y),len(x)])
    phase = np.zeros([len(y),len(x)])
    
    X, Y = np.meshgrid(x, y)
    
    w = 1/np.pi*np.arcsin(amplitude)
    p = 1/np.pi*phase
    
    hologram = 1/2+1/2*np.sign(np.cos(2*np.pi*X/x0+2*np.pi*Y/y0+np.pi*p) - np.cos(np.pi*w))
    
    if centralCircle == 1:
    
        for i in range(len(y)):
            
            for j in range(len(x)):
                
                if np.sqrt((y[i]/verticalFactor)**2+(x[j]/horzScaleFactor)**2) >= radius:
                    
                    hologram[i,j] = 0
    
    if save == 1:
        plt.gray()
        matplotlib.image.imsave(os.path.join(os.getcwd(),"tempFile.png"), hologram)
        plt.close()
        img = Image.open(os.path.join(os.getcwd(),"tempFile.png")).convert("RGB")
        img.save(os.path.join(os.getcwd(), saveFileName+".bmp"))
        os.remove(os.path.join(os.getcwd(),"tempFile.png"))
    
    return hologram

def leeLaguerreHologram(l,p,wl,w0,z,x,y,x0,y0,horzScaleFactor = 1,vertScaleFactor = 1,waistGaussian = 0,save = 0,saveFileName = 'leeLaguerreHologram'):
    
    """
    Parameters
    ----------
    l : int
        Order of the Hermite-Gaussian function. Must be equal or greater than 0.
    p : int
        Order of the Hermite-Gaussian function. Must be equal or greater than 0.
    wl : float
        Wavelength. Must be greater than 0. [m]
    w0 : float
        Beam waist. Must be greater than 0. [m]
    z : float
        Position on the optical axis. [m]
    x : array of floats
        Array with the x coordinate points to be considered. [m]
    y : array of floats
        Array with the x coordinate points to be considered. [m]
    x0 : float
        Carrier frequency wavelength along the x direction. Must be greater than 0.
    y0 : float
        Carrier frequency wavelength along the y direction. Must be greater than 0.
    horzScaleFactor : float, optional
        Scale factor to be applied on the horizontal direction. The default is 1.
    vertScaleFactor : float, optional
        Scale factor to be applied on the vertical direction. The default is 1.
    save : TYPE, optional
        If equals to 1, the hologram is saved to disc. The default is 0.
    saveFile : TYPE, optional
        Name of the file to be saved. The default is 'leeHermiteHologram.png'.

    Returns
    -------
    hologram : array
        Array containing the Lee hologram.

    """
    
    X, Y = np.meshgrid(x/horzScaleFactor, y/vertScaleFactor)
    
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
    
    if waistGaussian > 0:
    
        amplitudeGaussian = amplitudeGaussianLaguerre(x,y,z,wl,waistGaussian,horzScaleFactor,vertScaleFactor)
        amplitude = amplitude/amplitudeGaussian
        ampAbsNorm = abs(amplitude)/np.max(abs(amplitude))
        
    else:
        
        ampAbsNorm = abs(amplitude)/np.max(abs(amplitude))
    
    if z != 0:
        phase = (-k*r**2)/(2*R) + Phi + l*phi - k*z -  (np.sign(np.sign(amplitude)+0.1)-1)*(np.pi/2)
    else:
        phase =  (Phi + l*phi) - k*z - (np.sign(np.sign(amplitude)+0.1)-1)*(np.pi/2)
        
    ampAbsNorm = abs(amplitude)/np.max(abs(amplitude))

    w = 1/np.pi*np.arcsin(ampAbsNorm)
    p = 1/np.pi*phase
    
    hologram = 1/2+1/2*np.sign(np.cos(2*np.pi*X/x0+2*np.pi*Y/y0+np.pi*p) - np.cos(np.pi*w))
    
    if save == 1:
        plt.gray()
        matplotlib.image.imsave(os.path.join(os.getcwd(),"tempFile.png"), hologram)
        plt.close()
        img = Image.open(os.path.join(os.getcwd(),"tempFile.png")).convert("RGB")
        img.save(os.path.join(os.getcwd(),saveFileName+str(l)+str(p)+".bmp"))
        os.remove(os.path.join(os.getcwd(),"tempFile.png"))
    
    return hologram

def leeHermiteHologram(n,m,wl,w0,z,x,y,x0,y0,theta = 0,horzScaleFactor = 1,vertScaleFactor = 1,waistGaussian = 0,save = 0, saveFileName = 'leeHermiteHologram'):
    
    """
    Parameters
    ----------
    n : int
        Order of the Hermite-Gaussian function. Must be equal or greater than 0.
    m : int
        Order of the Hermite-Gaussian function. Must be equal or greater than 0.
    wl : float
        Wavelength. Must be greater than 0. [m]
    w0 : float
        Beam waist. Must be greater than 0. [m]
    z : float
        Position on the optical axis. [m]
    x : array of floats
        Array with the x coordinate points to be considered. [m]
    y : array of floats
        Array with the x coordinate points to be considered. [m]
    x0 : float
        Carrier frequency wavelength along the x direction. Must be greater than 0.
    y0 : float
        Carrier frequency wavelength along the y direction. Must be greater than 0.
    theta : float, optional
        Rotation angle in radians of the amplitude. The default is 0.
    horzScaleFactor : float, optional
        Scale factor to be applied on the horizontal direction. The default is 1.
    vertScaleFactor : float, optional
        Scale factor to be applied on the vertical direction. The default is 1.
    save : TYPE, optional
        If equals to 1, the hologram is saved to disc. The default is 0.

    Returns
    -------
    hologram : array
        Array containing the Lee hologram.

    """
    
    assert n >= 0 , "n must be equal or greater than 0."
    assert m >= 0 , "m must be equal or greater than 0."
    assert type(n) == int , "n must be an integer"
    assert type(m) == int , "m must be an integer"
    assert wl > 0 , "wl must greater than 0."
    assert w0 > 0 , "w0 must greater than 0."
    assert x0 > 0 , "x0 must greater than 0."
    assert y0 > 0 , "y0 must greater than 0."
    
    Xtemp, Ytemp = np.meshgrid(x, y)
    X = (Xtemp*np.cos(theta) - Ytemp*np.sin(theta))/horzScaleFactor
    Y = (Xtemp*np.sin(theta) + Ytemp*np.cos(theta))/vertScaleFactor
    
    k = (2*np.pi)/wl
    z_r = (k*w0**2)/2
    w = w0*np.sqrt(1 + z**2/z_r**2)
    H_n = hermite(n, monic = True)
    H_m = hermite(m, monic = True)

    if z != 0:
        R = (z_r**2 + z**2)/z
        
    amplitude = (w0/w)*np.exp(-(X**2+Y**2)/w**2)*H_m(np.sqrt(2)*X/w)*H_n(np.sqrt(2)*Y/w)
    
    if waistGaussian > 0:
    
        amplitudeGaussian = amplitudeGaussianHermite(x,y,z,wl,waistGaussian,horzScaleFactor,vertScaleFactor)
        amplitude = amplitude/amplitudeGaussian
        ampAbsNorm = abs(amplitude)/np.max(abs(amplitude))
        
    else:
        
        ampAbsNorm = abs(amplitude)/np.max(abs(amplitude))

    if z != 0:
        phase = (-k*(X**2+Y**2))/(2*R) + (1+n+m)*np.arctan2(z,z_r) - k*z - (np.sign(np.sign(amplitude)+0.1)-1)*(np.pi/2)
    else:
        phase =  (1+n+m)*np.arctan2(z,z_r) - k*z -  (np.sign(np.sign(amplitude)+0.1)-1)*(np.pi/2)
        
    w = 1/np.pi*np.arcsin(ampAbsNorm)
    p = 1/np.pi*phase

    hologram = 1/2+1/2*np.sign(np.cos(2*np.pi*Y/x0+2*np.pi*Y/y0+np.pi*p) - np.cos(np.pi*w))
    
    if save == 1:
        plt.gray()
        matplotlib.image.imsave(os.path.join(os.getcwd(),"tempFile.png"), hologram)
        plt.close()
        img = Image.open(os.path.join(os.getcwd(),"tempFile.png")).convert("RGB")
        img.save(os.path.join(os.getcwd(),saveFileName+str(n)+str(m)+".bmp"))
        os.remove(os.path.join(os.getcwd(),"tempFile.png"))
    
    return hologram

x0 = 10.8e-6*10000000000
y0 = 10.8e-6*20
x = np.linspace(0,10.8e-6*911,912)-10.8e-6*911/2
y = np.linspace(0,10.8e-6*1139,1140)-10.8e-6*1139/2
n = 0
m = 1
wl = 1550e-9
z = 0
horzScaleFactor = 1/2
waistGaussian = 2e-3
save = 1


w0s = np.array([0.5,1,1.5,1.9])*1e-3
name = [5,10,15,19]

for i in range(len(w0s)):  
    saveFileName = 'leeHermite'+str(name[i])+'mm'
    hologram = leeHermiteHologram(n,m,wl,w0s[i],z,x,y,x0,y0,horzScaleFactor = horzScaleFactor,waistGaussian = waistGaussian,save = save, saveFileName = saveFileName)
plt.imshow(hologram)
plt.gca().set_aspect('equal')
