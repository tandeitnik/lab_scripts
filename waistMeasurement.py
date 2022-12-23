#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:10:53 2022

@author: tandeitnik
"""

import numpy as np
from scipy.optimize import curve_fit

z = np.linspace(0, 12,13)*1e-3
hPositions = np.array([5.975,
                       5.970,
                       5.960,
                       5.952,
                       5.942,
                       5.938,
                       5.922,
                       5.910,
                       5.900,
                       5.890,
                       5.880,
                       5.870,
                       5.860
                       ])*1e-3

lPositions = np.array([3.295,
                       3.290,
                       3.285,
                       3.280,
                       3.272,
                       3.270,
                       3.260,
                       3.252,
                       3.248,
                       3.240,
                       3.235,
                       3.230,
                       3.225])*1e-3

radii = hPositions - lPositions

l = 780e-9 #wavelength

def model(z,a,b):
    
    return a*z +b

fit = curve_fit(model,z,radii)
ans, cov = fit

w0 = l/(abs(ans[0])*np.pi)
w0err = np.sqrt(cov[1,1])
