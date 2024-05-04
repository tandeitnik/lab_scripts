from bitPython import biquadFilter as bf
import numpy as np
import os

fs = 125e6/2**3 
omega0 = 0*2*np.pi 

a,b = bf.identity()

gain_a, gain_b = bf.fpgaCoef(a,b)

stringConfig = "1 1 " + str(gain_a[0]) + " " +str(gain_a[1])+ " "+str(gain_b[0])+" "+str(gain_b[1])+" "+str(gain_b[2])+" "+ str(gain_a[0]) + " " +str(gain_a[1])+ " "+str(gain_b[0])+" "+str(gain_b[1])+" "+str(gain_b[2])
               

os.system("ssh root@rp-f0b423.local \"/opt/scripts/biquadConfig "+stringConfig+" \"")