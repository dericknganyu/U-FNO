import numpy as np
import matplotlib.pyplot as plt
import time
import os
import h5py
from generateData import gen_data
from colorMap import parula

def readtoArray(hfile, N_train, N_test, Nx = 512, Ny = 512, deleteFile = False):
    
    print('Reading file..')
    t = time.time()
    hf = h5py.File(hfile, "r")
    F_train = hf['train_in']
    F_test  = hf['test_in']
    U_train = hf['train_out']
    U_test  = hf['test_out']
    
    print('     Reading file Complete after %s!'%(time.time() - t))
    print()
    print("     Dataset Information:")
    print("     ",F_train)
    print("     ",U_train)
    print("     ",F_test)
    print("     ",U_test)
    
    print()
    
    print('     Expected: N_train = %s, N_test = %s, Resolution = %s X %s'%(N_train, N_test, Nx+1, Ny+1))
    print('     Read:     N_train = %s, N_test = %s, Resolution = %s X %s'%(F_train.shape[0], F_test.shape[0], F_train.shape[1], F_train.shape[2]))
    if deleteFile:
        os.remove(hfile)
    
    print() 

    return F_train, U_train, F_test, U_test
    


if __name__ == '__main__':
    xmin, xmax = 0, 1 #-1, 1
    ymin, ymax = 0, 1 #-1, 1
    Nx, Ny = 512, 512 #16, 16
    N_train, N_test = 1024, 5000
    
    prefix = "../../../../../../localdata/Derick/stuart_data/Darcy_421/"#"../../../../../../localdata/Derick/PCANN/"
    fileName = prefix+"fUG_Square_TrainData=%s_TestData=%s_Resolution=%sX%s_Domain=[%s,%s]X[%s,%s].hdf5"%(N_train, N_test, Nx+1, Ny+1, xmin, xmax, ymin, ymax)
    
    #fileName = gen_data(xmin, xmax, ymin, ymax, Nx, Ny, N_train, N_test)
    F_train, U_train, F_test, U_test= readtoArray(fileName, N_train, N_test, Nx, Ny)#, deleteFile = True)
    
    fig = plt.figure(figsize=(10, 5))
    
    colourMap = parula() #plt.cm.jet
                
    fig.suptitle("Plot of a randomly generated $f$ and its FDM generated $u$ satisfying $- \Delta u = f, \partial \Omega = 0$ on $\Omega = [%s,%s]x[%s,%s]$"%(xmin, xmax, ymin, ymax))
    
    plt.subplot(1, 2, 1)
    plt.xlabel('x')#, fontsize=16, labelpad=15)
    plt.ylabel('y')#, fontsize=16, labelpad=15)
    plt.title("f")
    plt.imshow(F_train[0], cmap=colourMap, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.xlabel('x')#, fontsize=16, labelpad=15)
    plt.ylabel('y')#, fontsize=16, labelpad=15)
    plt.title("u")
    plt.imshow(U_train[0], cmap=colourMap, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
    plt.colorbar()
    
    t = time.localtime()
    plt.savefig("figures/f_and_u_"+time.strftime('%Y%m%d-%H:%M:%S', t))
    plt.show()