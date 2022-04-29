# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:00:08 2022

@author: Labq
"""

# OscilloscopeStream.py
#
# This example performs a stream mode measurement and writes the data to OscilloscopeStream.csv.
#
# Find more information on http://www.tiepie.com/LibTiePie .

from __future__ import print_function
import time
import os
import sys
import libtiepie
from tqdm import tqdm
from printinfo import *

#name = input('z position')

sample_frequency = 1e4 #quantidade de pontos por traço
acq_time = 10 # tempo total do traço

freq = sample_frequency/acq_time


delay = acq_time*2
record_length = int(acq_time / (1/sample_frequency))
#record_length = 1000  # 1 kS
N = 20
path =  r'C:\Users\Labq\Dropbox\Daniel_RT\teste\data'  #diretorio para salvar


def getNextFilePath(output_folder):
    highest_num = 0
    for f in os.listdir(output_folder):
        #print(f)
        if os.path.isfile(os.path.join(output_folder, f)):
            file_name = os.path.splitext(f)[0]
            #print(file_name)
            try:
                file_num = int(file_name)
                #print(file_num)
                if file_num > highest_num:
                    highest_num = file_num
            except ValueError:
                'The file name "%s" is not an integer. Skipping' % file_name

    output_file = os.path.join(output_folder, str(highest_num + 1))
    #print(output_file)
    return output_file

for n in tqdm(range(N)):

    output_file = getNextFilePath(path)
    
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
            scp.sample_frequency = sample_frequency  # 1 kHz
    
            # Set record length:
            scp.record_length = record_length
    
            # For all channels:
            for ch in scp.channels:
                # Enable channel to measure it:
                ch.enabled = True
    
                # Set range:
                ch.range = 8  # 8 V
    
                # Set coupling:
                ch.coupling = libtiepie.CK_DCV  # DC Volt
    
            # Print oscilloscope info:
            #print_device_info(scp)
    
            # Start measurement:
            scp.start()
    
            csv_file = open(output_file+'.csv', 'w')
            try:
                # Write csv header:
                csv_file.write('Sample')
                for i in range(len(scp.channels)):
                    csv_file.write(';Ch' + str(i + 1))
                csv_file.write(os.linesep)
    
    
                # Measure 10 chunks:
                #print()
                sample = 0
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
                    m = print(len(data[1]))
    
                    # Output CSV data:
                    for i in range(len(data[0])):
                        csv_file.write(str(sample + i))
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

    time.sleep(delay)