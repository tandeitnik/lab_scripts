# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import os
import sys
import libtiepie
import beepy
from tqdm import tqdm
from printinfo import *

sample_frequency = 10_000 #quantidade de pontos por segundo
acq_time = 1 # tempo total do traço [s]
output_folder = r"C:\Users\Labq\Dropbox\scripts\tiepie\data" #pasta raíz aonde a nova pasta será criada
N = 100 # número de traços
#freq = sample_frequency/acq_time
freq = sample_frequency
dt = 1/freq

record_length = int(acq_time / (1/sample_frequency))
delay = 0.5

file_names = []

for i in range(N):
    
    if i <= 9:
        
        file_names.append('0'+str(i))
        
    else:
        
        file_names.append(str(i))

for n in tqdm(range(N)):

    output_file = os.path.join(output_folder,file_names[n])
    
    # Print library info:
    #print_library_info()
    
    # Enable network search:
    libtiepie.network.auto_detect_enabled = True
    
    # Search for devices:
    libtiepie.device_list.update()
    
    # Try to open an oscilloscope with stream measurement support:
    scp = None
    for item in libtiepie.device_list:
        if item.can_open(libtiepie.DEVICETYPE_OSCILLOSCOPE):
            scp = item.open_oscilloscope()
            if scp.measure_modes & libtiepie.MM_STREAM:
                break
            else:
                scp = None
    
    if scp:
        try:
            # Set measure mode:
            scp.measure_mode = libtiepie.MM_STREAM
    
            # Set sample frequency:
            scp.sample_frequency = sample_frequency
    
            # Set record length:
            scp.record_length = record_length
    
            # For all channels:
            for ch in scp.channels:
                # Enable channel to measure it:
                ch.enabled = True
    
                # Set range:
                ch.range = 10  # 8 V
    
                # Set coupling:
                ch.coupling = libtiepie.CK_DCV
    
            # Print oscilloscope info:
            #print_device_info(scp)
    
            # Start measurement:
            scp.start()
    
            csv_file = open(output_file+'.csv', 'w')
            try:
                # Write csv header:
                csv_file.write('Sample')
                csv_file.write(';Rel. Time')
                for i in range(len(scp.channels)):
                    csv_file.write(';Ch' + str(i + 1))
                csv_file.write(os.linesep)

                sample = 0
                relativeTime = 0
                for chunk in range(1):
                    # Print a message, to inform the user that we still do something:
                    #print('Data chunk ' + str(chunk + 1))
    
                    # Wait for measurement to complete:
                    while not (scp.is_data_ready or scp.is_data_overflow):
                        time.sleep(0.01)  # 10 ms delay, to save CPU time
    
                    if scp.is_data_overflow:
                        print('Data overflow!')
                        break
    
                    # Get data:
                    data = scp.get_data()
                    #m = print(len(data[1]))
    
                    # Output CSV data:
                    for i in range(len(data[0])):
                        csv_file.write(str(sample + i))
                        csv_file.write(';' + str(relativeTime+i*dt))
                        for j in range(len(data)):
                            csv_file.write(';' + str(data[j][i]))
                        csv_file.write(os.linesep)
    
                    sample += len(data[0])
    
                #print()
                #print('Data written to: ' + csv_file.name)
            finally:
                csv_file.close()
    
            # Stop stream:
            scp.stop()
    
        except Exception as e:
            print('Exception: ' + e.message)
            sys.exit(1)
    
        # Close oscilloscope:
        del scp
    
    else:
        print('No oscilloscope available with stream measurement support!')
        sys.exit(1)

beepy.beep(sound=1)
