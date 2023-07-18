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


def leeLaguerreHologram(l,p,wl,w0,z,x,y,x0,y0,horzScaleFactor = 1,vertScaleFactor = 1,save = 0,saveFile = 'leeLaguerreHologram.png'):
    
    """
    Parameters
    ----------
    l : int
        Order of the Hermite-Gaussian function. Must be equal or greater than 0.
    p : int
        Order of the Hermite-Gaussian function. Must be equal or greater than 0.
    wl : float
        Wavelength. Must be greater than 0.
    w0 : float
        Beam waist. Must be greater than 0.
    z : float
        Position on the optical axis.
    x : array of floats
        Array with the x coordinate points to be considered.
    y : array of floats
        Array with the x coordinate points to be considered.
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
        img.save(os.path.join(os.getcwd(),"laguerre"+str(l)+str(p)+".bmp"))
        os.remove(os.path.join(os.getcwd(),"tempFile.png"))
    
    return hologram

def leeHermiteHologram(n,m,wl,w0,z,x,y,x0,y0,theta = 0,horzScaleFactor = 1,vertScaleFactor = 1,save = 0):
    
    """
    Parameters
    ----------
    n : int
        Order of the Hermite-Gaussian function. Must be equal or greater than 0.
    m : int
        Order of the Hermite-Gaussian function. Must be equal or greater than 0.
    wl : float
        Wavelength. Must be greater than 0.
    w0 : float
        Beam waist. Must be greater than 0.
    z : float
        Position on the optical axis.
    x : array of floats
        Array with the x coordinate points to be considered.
    y : array of floats
        Array with the x coordinate points to be considered.
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

    if z != 0:
        phase = (-k*(X**2+Y**2))/(2*R) + (1+n+m)*np.arctan2(z,z_r) - k*z - (np.sign(np.sign(amplitude)+0.1)-1)*(np.pi/2)
    else:
        phase =  (1+n+m)*np.arctan2(z,z_r) - k*z -  (np.sign(np.sign(amplitude)+0.1)-1)*(np.pi/2)
        
    ampAbsNorm = abs(amplitude)/np.max(abs(amplitude))

    w = 1/np.pi*np.arcsin(ampAbsNorm)
    p = 1/np.pi*phase

    hologram = 1/2+1/2*np.sign(np.cos(2*np.pi*Y/x0+2*np.pi*Y/y0+np.pi*p) - np.cos(np.pi*w))
    
    if save == 1:
        plt.gray()
        matplotlib.image.imsave(os.path.join(os.getcwd(),"tempFile.png"), hologram)
        plt.close()
        img = Image.open(os.path.join(os.getcwd(),"tempFile.png")).convert("RGB")
        img.save(os.path.join(os.getcwd(),"hermite"+str(n)+str(m)+".bmp"))
        os.remove(os.path.join(os.getcwd(),"tempFile.png"))
    
    return hologram
