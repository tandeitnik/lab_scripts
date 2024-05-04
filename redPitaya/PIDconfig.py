from bitPython import bitPython as bp
from bitPython import biquadFilter as bf
import numpy as np
import os

#########################
##GENERAL CONFIGURATION##
#########################

rp_mask = "root@rp-f0b423.local"
decimator = 3 #power of two
rp_freq = 125e6 # [Hz] red pitaya clock

########################
##FILTER CONFIGURATION##
########################

enableFilter = 1

filterType = "identity" # "identity", "low pass", "high pass", "band pass", "notch"
order = 1 #only valid for low and high pass
cornerFreq = 0 #[Hz]
freqWidth  = 0 #[Hz]

#####################
##PID CONFIGURATION##
#####################

enablePID = 1

setPointOrigin = "RAM" # "RAM", "ADC"
setPoint = 0.5 #[V]

clampLowBound = -0.85 #[V]
clampHighBound = 0.85 #[V]

P = 10
I = 1 #units of sec-1
D = 0 #units of sec


def makeStringFilter(filterType,order,rp_freq,decimator,cornerFreq,freqWidth):
    
    fs = rp_freq/(2**decimator)
    
    if filterType == "identity":
        a,b = bf.identity()
    elif filterType == "low pass":
        a,b = bf.lowPassCoef(fs,cornerFreq*2*np.pi,int(order))
    elif filterType == "high pass":
        a,b = bf.highPassCoef(fs,cornerFreq*2*np.pi,int(order))
    elif filterType == "band pass":
        a,b = bf.bandPassCoef(fs,cornerFreq*2*np.pi,freqWidth*2*np.pi)
    elif filterType == "notch":
        a,b = bf.notchCoef(fs,cornerFreq*2*np.pi,freqWidth*2*np.pi)
        
    gain_a, gain_b = bf.fpgaCoef(a,b)
    
    string = " "+str(gain_a[0]) +" "+str(gain_a[1])+ " "+str(gain_b[0])+" "+str(gain_b[1])+" "+str(gain_b[2])
    
    return string

def makeStringPID(setPoint,P,I,D,clampLowBound,clampHighBound):
    
    setPoint_RP = bp.fixed2dec(setPoint,1,13)
    low_bound_RP = bp.fixed2dec(clampLowBound,1,13)
    high_bound_RP = bp.fixed2dec(clampHighBound,1,13)
    Kp_RP = bp.fixed2dec(P,16,16)
    Ki_RP = bp.fixed2dec(I,16,16)
    Kd_RP = bp.fixed2dec(D,16,16)
    
    string = " "+str(setPoint_RP)+" "+str(low_bound_RP)+" "+str(high_bound_RP)+" "+str(Kp_RP)+" "+str(Ki_RP)+" "+str(Kd_RP)
    
    return string


stringFilter = makeStringFilter(filterType,order,rp_freq,decimator,cornerFreq,freqWidth)
stringPID = makeStringPID(setPoint,P,I,D,clampLowBound,clampHighBound)

if setPointOrigin == 'RAM':
    enableString = str(int(enableFilter))+" "+str(int(enablePID))+" 0"
elif setPointOrigin == 'ADC':
    enableString = str(int(enableFilter))+" "+str(int(enablePID))+" 1"
    
stringConfig = enableString + stringFilter + stringPID

os.system("ssh "+rp_mask+" \"/opt/scripts/PIDconfig "+stringConfig+" \"")
