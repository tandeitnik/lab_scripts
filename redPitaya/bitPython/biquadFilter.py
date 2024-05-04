import numpy as np
import bitPython as bp

def identity():

    a = [0,0,0]
    b = [1,0,0]
    
    return a,b

def lowPassCoef(fs,omega0,order):
    
    dt = 1/fs
    alpha = omega0*dt
    a = [-1,0,0]
    b = [0,0,0]
    
    if order == 1:
        
        a[1] = -(alpha - 2.0)/(alpha+2.0)
        b[0] = alpha/(alpha+2.0)
        b[1] = alpha/(alpha+2.0)
        
    elif order == 2:
        
        D = alpha**2 + 2*np.sqrt(2)*alpha + 4
        a[1] = (8-2*alpha**2)/D
        a[2] = (-alpha**2 + 2*np.sqrt(2)*alpha - 4)/D
        b[0] = alpha**2/D
        b[1] = 2*alpha**2/D
        b[2] = alpha**2/D
        
    return a,b

def highPassCoef(fs,omega0,order):
    
    dt = 1/fs
    alpha = omega0*dt
    a = [-1,0,0]
    b = [0,0,0]
    
    if order == 1:
        
        a[1] = -(alpha/2-1)/(1+alpha/2)
        b[0] = 1/(1+alpha/2)
        b[1] = -1/(1+alpha/2)
        
    elif order == 2:
        
        D = alpha**2 + 2*np.sqrt(2)*alpha + 4
        a[1] = (8-2*alpha**2)/D
        a[2] = (-alpha**2 + 2*np.sqrt(2)*alpha - 4)/D
        b[0] = 4/D
        b[1] = -8/D
        b[2] = 4/D
        
    return a,b

def bandPassCoef(fs,omega0,dw):
    
    dt = 1/fs
    alpha = omega0*dt
    Q = omega0/dw
    a = [-1,0,0]
    b = [0,0,0]
        
    D = 2*alpha + 4*Q + Q*alpha**2
    a[1] = (8*Q-2*Q*alpha**2)/D
    a[2] = (2*alpha-4*Q-Q*alpha**2)/D
    b[0] = 2*alpha/D
    b[1] = 0
    b[2] = -2*alpha/D
        
    return a,b

def notchCoef(fs,omega0,dw):
    
    dt = 1/fs
    alpha = omega0*dt
    Q = omega0/dw
    a = [-1,0,0]
    b = [0,0,0]
        
    D = 2*alpha + 4*Q + Q*alpha**2
    a[1] = (8*Q-2*Q*alpha**2)/D
    a[2] = (2*alpha-4*Q-Q*alpha**2)/D
    b[0] = (4*Q+alpha**2*Q)/D
    b[1] = (-8*Q+2*alpha**2*Q)/D
    b[2] = (4*Q+alpha**2*Q)/D
        
    return a,b

def fpgaCoef(a,b):
    
    gain_a = [0,0]
    gain_b = [0,0,0]
    
    gain_a[0] = bp.fixed2dec(a[1],4,28)
    gain_a[1] = bp.fixed2dec(a[2],4,28)
    
    gain_b[0] = bp.fixed2dec(b[0],4,28)
    gain_b[1] = bp.fixed2dec(b[1],4,28)
    gain_b[2] = bp.fixed2dec(b[2],4,28)

    return gain_a, gain_b
        
def biquadFilter(a,b,signal):
    
    filteredSignal = np.zeros(len(signal))
    
    for i in range(2,len(signal)):
        
        filteredSignal[i] = a[1]*filteredSignal[i-1] + \
                            a[2]*filteredSignal[i-2] + \
                            b[0]*signal[i] + \
                            b[1]*signal[i-1] + \
                            b[2]*signal[i-2]
        
    return filteredSignal
    


