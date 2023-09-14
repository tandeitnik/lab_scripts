# -*- coding: utf-8 -*-
"""
@author: tandeitnik
"""
import numpy as np
from scipy.special import hermite
from scipy.special import eval_genlaguerre
import matplotlib.pyplot as plt
import matplotlib.image
from PIL import Image
import os

def cylindricalTEM(p,l,wl,w0,z,x,y,theta = 0):
    """
    Generates the complex amplitude of a cylindricam TEM

    Parameters
    ----------
    p : int
        radial index.
    l : int
        azimuthal index.
    wl : float
        wavelength [m].
    w0 : float
        waist [m].
    z : float
        position along the optical axis [m].
    x : float array
        arrays of positions along the transversal x direction [m].
    y : float array
        arrays of positions along the transversal y direction [m].
    theta : float, optional
        orientation angle [degrees]. The default is 0.

    Returns
    -------
    U : complex
        normalized complex amplitude.

    """
    
    Xtemp, Ytemp = np.meshgrid(x, y)
    X = (Xtemp*np.cos(np.radians(theta)) - Ytemp*np.sin(np.radians(theta)))
    Y = (Xtemp*np.sin(np.radians(theta)) + Ytemp*np.cos(np.radians(theta)))
    
    r = np.sqrt(X**2+Y**2)
    phi = np.arctan2(Y,X)
    
    k = (2*np.pi)/wl
    z_r = (k*w0**2)/2
    w = w0*np.sqrt(1 + z**2/z_r**2)
    laguerre = eval_genlaguerre(p, abs(l), 2*r**2/w**2)
    
    if z != 0:
        R = (z_r**2 + z**2)/z
        
    Phi = (2*p+abs(l)+1)*np.arctan2(z,z_r)
    
    amplitude = (1/w)*np.exp(((-r**2)/w**2))*((np.sqrt(2)*r/w)**abs(l))*laguerre*2*np.cos(l*phi)
    
    if z != 0:
        phase = (-k*r**2)/(2*R) + Phi  -  (np.sign(np.sign(amplitude)+0.1)-1)*(np.pi/2)
    else:
        phase =  Phi - (np.sign(np.sign(amplitude)+0.1)-1)*(np.pi/2)
        
    U = abs(amplitude)*np.exp(1j*phase)
    
    U = U/np.max(abs(amplitude))
    
    return U

def hermiteGauss(n,m,wl,w0,z,x,y,theta = 0):
    """
    Generates the complex amplitude of a hermite-gaussian beam
    
    Parameters
    ----------
    n : int
        n index.
    m : int
        m index.
    wl : float
        wavelength [m].
    w0 : float
        waist [m].
    z : float
        position along the optical axis [m].
    x : float array
        arrays of positions along the transversal x direction [m].
    y : float array
        arrays of positions along the transversal y direction [m].
    theta : float, optional
        orientation angle [degrees]. The default is 0.

    Returns
    -------
    U : complex
        normalized complex amplitude.

    """
    
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
    
    if z != 0:
        phase = (-k*(X**2+Y**2))/(2*R) + (1+n+m)*np.arctan2(z,z_r) - k*z - (np.sign(np.sign(amplitude)+0.1)-1)*(np.pi/2)
    else:
        phase =  (1+n+m)*np.arctan2(z,z_r) - k*z -  (np.sign(np.sign(amplitude)+0.1)-1)*(np.pi/2)
        
    U = abs(amplitude)*np.exp(1j*phase)
    
    U = U/np.max(abs(amplitude))
    
    
    return U

def laguerreGauss(p,l,wl,w0,z,x,y):
    """
    Generates the complex amplitude of a laguerre-gaussian beam

    Parameters
    ----------
    p : int
        radial index.
    l : int
        azimuthal index.
    wl : float
        wavelength [m].
    w0 : float
        waist [m].
    z : float
        position along the optical axis [m].
    x : float array
        arrays of positions along the transversal x direction [m].
    y : float array
        arrays of positions along the transversal y direction [m].

    Returns
    -------
    U : complex
        normalized complex amplitude.

    """
    
    X, Y = np.meshgrid(x, y)
    
    r = np.sqrt(X**2+Y**2)
    phi = np.arctan2(Y,X)
    

    k = (2*np.pi)/wl
    z_r = (k*w0**2)/2
    w = w0*np.sqrt(1 + z**2/z_r**2)
    laguerre = eval_genlaguerre(p, abs(l), 2*r**2/w**2)
    
    if z != 0:
        R = (z_r**2 + z**2)/z
        
    Phi = (2*p+abs(l)+1)*np.arctan2(z,z_r)
    
    amplitude = (1/w)*np.exp(((-r**2)/w**2))*((np.sqrt(2)*r/w)**abs(l))*laguerre

    
    if z != 0:
        phase = (-k*r**2)/(2*R) + Phi + l*phi - k*z -  (np.sign(np.sign(amplitude)+0.1)-1)*(np.pi/2)
    else:
        phase =  (Phi + l*phi) - k*z - (np.sign(np.sign(amplitude)+0.1)-1)*(np.pi/2)
        
    U = abs(amplitude)*np.exp(1j*phase)
    
    U = U/np.max(abs(amplitude))
    
    return U
    
def correctionGaussian(x,y,z,wl,w0,h = 0, v = 0, incidentAngle = 0):
    """
    generates the amplitude of a deformed Gaussian beam for correction

    Parameters
    ----------
    x : float array
        arrays of positions along the transversal x direction [m].
    y : float array
        arrays of positions along the transversal y direction [m].
    z : float
        position along the optical axis [m].
    wl : float
        wavelength [m].
    w0 : float
        waist [m].
    h : float, optional
        horizontal displacement correction [m]. The default is 0.
    v : float, optional
        vertical displacement correction [m]. The default is 0.
    incidentAngle : float, optional
        incident angle on the DMD [degrees]. The default is 0.

    Returns
    -------
    amplitude : float

    """

    correctionFactor = 1/np.cos(np.radians(incidentAngle))
    X, Y = np.meshgrid((x-h)/correctionFactor, y-v)
    
    r = np.sqrt(X**2+Y**2)
    
    k = (2*np.pi)/wl
    z_r = (k*w0**2)/2
    w = w0*np.sqrt(1 + z**2/z_r**2)
    
    amplitude = (1/w)*np.exp(((-r**2)/w**2))
    amplitude = amplitude/np.max(amplitude)
    
    return amplitude

def TEMhologram(p,l,wl,w0,z,x,y,x0,y0,theta = 0,h = 0,v = 0,waistGaussian = 0,incidentAngle = 0,invert = 0, lee = 0, save = 0, saveFileName = 'ditTEMHologram'):
    """
    generates an off-axis dithered hologram for a TEM mode

    Parameters
    ----------
    p : int
        radial index.
    l : int
        azimuthal index.
    wl : float
        wavelength [m].
    w0 : float
        waist [m].
    z : float
        position along the optical axis [m].
    x : float array
        arrays of positions along the transversal x direction [m].
    y : float array
        arrays of positions along the transversal y direction [m].
    x0 : float
        x axis spatial period of the carrier plane wave.
    y0 : float
        y axis spatial period of the carrier plane wave..
    theta : float, optional
       orientation angle [degrees]. The default is 0.
    h : float, optional
        horizontal displacement correction [m]. The default is 0.
    v : float, optional
        vertical displacement correction [m]. The default is 0.
    waistGaussian : float, optional
        waist of the incoming Gaussian beam [m]. The default is 0.
    incidentAngle : float, optional
        incidente angle of the incoming Gaussian beam [degrees]. The default is 0.
    invert : int, optional
        if 1, inverts white and black. The default is 0.
    lee : int, optional
        if 1, makes a Lee hologram. The default is 0.
    save : int, optional
        if 1, saves the hologram to disk. The default is 0.
    saveFileName : string, optional
        name of the file to be saved. The default is 'ditHermiteHologram'.

    Returns
    -------
    hologram : float array

    """
    
    U = cylindricalTEM(p,l,wl,w0,z,x,y,theta = 0)
    
    X, Y = np.meshgrid(x, y)
    
    if waistGaussian != 0:
        
        amplitudeGaussian = correctionGaussian(x,y,z,wl,w0 = waistGaussian,h = h,v = v, incidentAngle = incidentAngle)
        U = U/amplitudeGaussian
    
    if lee == 1:
        p = np.angle(U)
        q = np.arcsin(np.abs(U))
        hologram = 1/2+1/2*np.sign(np.cos(2*np.pi*X/x0+2*np.pi*Y/y0 + p) - np.cos(q))      
    else:
        phasePlane = np.exp(1j*(2*np.pi*X/x0 + 2*np.pi*Y/y0))
        hologram = np.abs(U+phasePlane)**2
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
    
def hermiteHologram(n,m,wl,w0,z,x,y,x0,y0,theta = 0,h = 0,v = 0,waistGaussian = 0,incidentAngle = 0,invert = 0, lee = 0, save = 0, saveFileName = 'ditHermiteHologram'):
    """
    generates an off-axis dithered hologram for a Hermite-Gaussian mode

    Parameters
    ----------
    n : int
        n index.
    m : int
        m index.
    wl : float
        wavelength [m].
    w0 : float
        waist [m].
    z : float
        position along the optical axis [m].
    x : float array
        arrays of positions along the transversal x direction [m].
    y : float array
        arrays of positions along the transversal y direction [m].
    x0 : float
        x axis spatial period of the carrier plane wave.
    y0 : float
        y axis spatial period of the carrier plane wave..
    theta : float, optional
       orientation angle [degrees]. The default is 0.
    h : float, optional
        horizontal displacement correction [m]. The default is 0.
    v : float, optional
        vertical displacement correction [m]. The default is 0.
    waistGaussian : float, optional
        waist of the incoming Gaussian beam [m]. The default is 0.
    incidentAngle : float, optional
        incidente angle of the incoming Gaussian beam [degrees]. The default is 0.
    invert : int, optional
        if 1, inverts white and black. The default is 0.
    lee : int, optional
        if 1, makes a Lee hologram. The default is 0.
    save : int, optional
        if 1, saves the hologram to disk. The default is 0.
    saveFileName : string, optional
        name of the file to be saved. The default is 'ditHermiteHologram'.

    Returns
    -------
    hologram : float array

    """
    
    U = hermiteGauss(n,m,wl,w0,z,x,y,theta)
    
    Xtemp, Ytemp = np.meshgrid(x, y)
    X = (Xtemp*np.cos(np.radians(theta)) - Ytemp*np.sin(np.radians(theta)))
    Y = (Xtemp*np.sin(np.radians(theta)) + Ytemp*np.cos(np.radians(theta)))
    
    
    if waistGaussian != 0:
        
        amplitudeGaussian = correctionGaussian(x,y,z,wl,w0 = waistGaussian,h = h,v = v, incidentAngle = incidentAngle)
        U = U/amplitudeGaussian
        
    if lee == 1:
        p = np.angle(U)
        q = np.arcsin(np.abs(U))
        hologram = 1/2+1/2*np.sign(np.cos(2*np.pi*X/x0+2*np.pi*Y/y0 + p) - np.cos(q))      
    else:
        phasePlane = np.exp(1j*(2*np.pi*X/x0 + 2*np.pi*Y/y0))
        hologram = np.abs(U+phasePlane)**2
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

def laguerreHologram(p,l,wl,w0,z,x,y,x0,y0,h = 0,v = 0,waistGaussian = 0,incidentAngle = 0,invert = 0,lee = 0, save = 0, saveFileName = 'ditLaguerreHologram'):
    """
    generates an off-axis dithered hologram for a Laguerre-Gaussian mode

    Parameters
    ----------
    p : int
        radial index.
    l : int
        azimuthal index.
    wl : float
        wavelength [m].
    w0 : float
        waist [m].
    z : float
        position along the optical axis [m].
    x : float array
        arrays of positions along the transversal x direction [m].
    y : float array
        arrays of positions along the transversal y direction [m].
    x0 : float
        x axis spatial period of the carrier plane wave.
    y0 : float
        y axis spatial period of the carrier plane wave..
    h : float, optional
        horizontal displacement correction [m]. The default is 0.
    v : float, optional
        vertical displacement correction [m]. The default is 0.
    waistGaussian : float, optional
        waist of the incoming Gaussian beam [m]. The default is 0.
    incidentAngle : float, optional
        incidente angle of the incoming Gaussian beam [degrees]. The default is 0.
    invert : int, optional
        if 1, inverts white and black. The default is 0.
    lee : int, optional
        if 1, makes a Lee hologram. The default is 0.
    save : int, optional
        if 1, saves the hologram to disk. The default is 0.
    saveFileName : string, optional
        name of the file to be saved. The default is 'ditHermiteHologram'.

    Returns
    -------
    hologram : float array

    """
    
    U = laguerreGauss(p,l,wl,w0,z,x,y)
    
    X, Y = np.meshgrid(x, y)
    
    
    if waistGaussian != 0:
        
        amplitudeGaussian = correctionGaussian(x,y,z,wl,w0 = waistGaussian,h = h,v = v, incidentAngle = incidentAngle)
        U = U/amplitudeGaussian
        
    if lee == 1:
        p = np.angle(U)
        q = np.arcsin(np.abs(U))
        hologram = 1/2+1/2*np.sign(np.cos(2*np.pi*X/x0+2*np.pi*Y/y0 + p) - np.cos(q))      
    else:
        phasePlane = np.exp(1j*(2*np.pi*X/x0 + 2*np.pi*Y/y0))
        hologram = np.abs(U+phasePlane)**2
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

def scaleConv(array,bits):
    """
    Normalizes an array to have values between 0 and 2**bits-1

    Parameters
    ----------
    array : float array
        array to be normalized.
    bits : int
        number of bits.

    Returns
    -------
    flaot
        Normalized array.

    """
    hValue = 2**bits - 1
    
    return (hValue/(np.max(array)-np.min(array)))*(array - np.min(array))

def floydDithering(image, bits, s = 0):
    """
    Apply Floyd dithering to a 2D array

    Parameters
    ----------
    image : float array

    bits : int
        number of bits of the values of the processed array.
    s : TYPE, optional
        if 1, it alternates de direction of the lines. The default is 0.

    Returns
    -------
    image : float array
        processd array.

    """
    
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
    """
    Apply Jarvis dithering to a 2D array

    Parameters
    ----------
    image : float array

    bits : int
        number of bits of the values of the processed array.
    s : TYPE, optional
        if 1, it alternates de direction of the lines. The default is 0.

    Returns
    -------
    image : float array
        processd array.

    """

    
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
