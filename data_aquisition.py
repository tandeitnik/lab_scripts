import numpy as np
import pyvisa as visa
import time
import os
from datetime import date
import tqdm
import beepy

N = 10  # Número de amostras
delay = 0.5  # Tempo entre as amostras, a unidade é segundos
experiment = '7' #número do experimento do dia

"""
É importante que o delay não seja 0 para que a aquisição ocorra corretamente!
O delay deve ser no mínimo o tempo total de cada amostra, mas preferencialmente
maior.
"""

today = date.today() #será usado para dar o nome à pasta
d = today.strftime("%b-%d-%Y")
root_folder = "C:/Users/Labq/Desktop/Felipe/python scripts" #pasta raíz aonde a nova pasta será criada
file_name = 'traces_CH2-CH3-CH4.npz' #nome do arquivo que será gravado

new_folder_name = 'exp_'+experiment+'_'+d #nome da nova pasta


def getData(inst,canal):    

    inst.write('DATa:SOUrce ' + canal)   
    
    inst.write("DATA:ENCDG RIBinary")
    
    inst.write('CURV?')
    
    data = inst.read_raw()
    # decode DATA
    n_ch = int(chr(data[1]))
    n_yy = int(data[2:2+n_ch])
    n_dw = int(inst.query("DATa:WIDth?"))
    
    data_n = []
    inicio = 2+n_ch
    for i in range(int(n_yy/n_dw)):
        if data[inicio] & 128:
            data_n.append(-(2**16) + (((data[inicio]) << 8) + data[inicio+1]))
        else:
            data_n.append((data[inicio] << 8) + data[inicio+1])
        inicio += 2
    data = data_n[:]
    del data_n
    
    inst.write('WFMP:'+canal+':YMULT?')           
    ymult = float(inst.read_raw())
    
    inst.write('WFMP:'+canal+':YUNIT?')            
    yunit = inst.read_raw().replace(b"\"", b"").replace(b"\n", b"").decode('utf8')
    
    inst.write('WFMP:'+canal+':YZERO?')           
    yzero = inst.read_raw()
    
    inst.write('WFMP:'+canal+':XINC?')           
    xinc = float(inst.read_raw())
    
    inst.write('WFMP:'+canal+':XUNIT?')           
    xunit = inst.read_raw().replace(b"\"", b"").replace(b"\n", b"").decode('utf8')
    
    inst.write('WFMP:'+canal+':NR_Pt?')           
    nr_pt = inst.read_raw()
    
    inst.write('WFMP:'+canal+':YOFF?') 
    yoff = float(inst.read_raw())

    inst.write('WFMPre:WFId?')
    axis = str(inst.read_raw())
    posX = axis.find('s/div')
    xaxis = float(axis[posX-7:posX-1])

    posY1 = axis.find('g,')
    posY2 = axis.find('V/div')
    yaxis = float(axis[posY1+3:posY2-1])
    
    x, y = [], []
    for i in range(len(data)):
        x.append(i*xinc)
        y.append((float(data[i])-yoff)*float(ymult))

    return x, y, xunit, yunit, xaxis, yaxis

rm = visa.ResourceManager()
lis = rm.list_resources()

inst = rm.open_resource('USB0::0x0699::0x036A::C045688::INSTR')

"""
Deixar não comentado abaixo os canais que serão coletados
"""
#traces_CH1 = [] #canal 1
traces_CH2 = [] #canal 2
traces_CH3 = [] #canal 3
traces_CH4 = [] #canal 4

test_samples = [0,0]

for i in range(2):
    inst.write("acq:state stop")
    test_samples[i] = tuple(getData(inst, 'CH2')[0:2])
    inst.write("acq:state run")
    time.sleep(delay)
    
if test_samples[0] == test_samples[1]:
    print("Os dados estão vindo iguais")
    
else:

    for i in range(N):
        inst.write("acq:state stop")
        #traces_CH1.append(np.array(getData(inst, 'CH1')[0:2]))
        traces_CH2.append(np.array(getData(inst, 'CH2')[0:2]))
        traces_CH3.append(np.array(getData(inst, 'CH3')[0:2]))
        traces_CH4.append(np.array(getData(inst, 'CH4')[0:2]))
        inst.write("acq:state run")
        print("Iteração", i+1, "de", N )
        time.sleep(delay)

        #creating new folder
    if not os.path.exists(root_folder+'/'+new_folder_name):
        os.makedirs(root_folder+'/'+new_folder_name)
    #saving traces to folder
    np.savez_compressed(root_folder+'/'+new_folder_name+'/'+file_name, traces_CH2, traces_CH3, traces_CH4)
    print("Dados salvos em "+root_folder+'/'+new_folder_name+'/'+file_name)

beepy.beep(sound=1)