# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:17:59 2024

@author: tandeitnik
"""

from bitPython import bitPython as bp
import os

#NCO parameters
fs = 125e6 #red pitaya clk frequency
f = 69_000 #NCO center frequency
phaseCorrection = (1/256)*0.01 # between 0 and 1, with 1 -> 2pi
freqCorrection = 1/fs
delay = 0.25 # between 0 and 1, with 1 -> 2pi

#comparator parameters
positiveOffset = 0  #V
negativeOffset = 0 #V


freq = bp.bit2decimal(bp.fixedPoint(f/fs,1,32)[1:])
freqCorrection = bp.bit2decimal(bp.fixedPoint(freqCorrection,1,32)[1:])
positiveOffset = bp.fixed2dec(positiveOffset,1,13)
negativeOffset = bp.fixed2dec(negativeOffset,1,13)
phaseCorrection = bp.bit2decimal(bp.fixedPoint(phaseCorrection*256,9,24)[1:])
delay = int(delay*256)

stringConfig = str(positiveOffset)+" "+str(negativeOffset)+" "+str(freq)+" "+str(phaseCorrection)+" "+str(delay)+" "+str(freqCorrection)+" 1"

os.system("ssh root@rp-f0b423.local \"/opt/scripts/PLLconfig "+stringConfig+" \"")
