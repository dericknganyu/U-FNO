import torch
import numpy as np
import matplotlib.pyplot as plt
from sample import sample_f
from poisson import poisson
import sys, os
import time
import h5py



def gen_data(xmin, xmax, ymin, ymax, Nx, Ny, N_train, N_test):
    startt = time.time()
    print()
    print("Data Generation Starts...")
    print()
    
    #Just something to prevent errors in the next set of code out of this file
    if N_train == 0:
        N_train = 1
    if N_test == 0:
        N_test = 1
    #End of Just something to prevent errors in the next set of code out of this file
                
    U = np.zeros((N_train + N_test, Ny+1,Nx+1))
    
    print ("     Generating Input Function...")
    t = time.time()
    FCN = sample_f(batch_size = N_train + N_test, n=Nx+1, m=Ny+1).numpy()
    print()
    print ("          Generation Completed after %s seconds"%(time.time() - t))
    print () 
    
    for i in range(N_train + N_test):
        print ("     %s of %s: Evaluating Output by FDM"%(i+1, N_train + N_test))
        t0 = time.time()
        U[i],X,Y = poisson(xmin, xmax, ymin, ymax, Nx, Ny, \
                        0.    , 0.    , 0.     , 0.   , FCN[i])
    #                   gNorth, gWest., gSouth., gEast, fcn)
    
        
        t1 = time.time()
        
        print ("          Evaluation Completed after %s seconds"%(t1 - t0))
        
    print()
    print ("     Now writing Dataset to file")
    prefix = ""#"../../../../../../localdata/Derick/PCANN/"
    fileName = prefix+"UG_Square_TrainData=%s_TestData=%s_Resolution=%sX%s_Domain=[%s,%s]X[%s,%s].hdf5"%(N_train, N_test, Nx+1, Ny+1, xmin, xmax, ymin, ymax)
    if os.path.isfile(fileName):
        os.remove(fileName)
        
    hf = h5py.File(fileName, "w")
    dset_train_in  = hf.create_dataset("train_in",  data = FCN[0:N_train], shape=(N_train, Nx+1, Ny+1), compression='gzip', chunks=True)
    print ("          Input Training data written!")
    
    dset_train_out = hf.create_dataset("train_out", data = U[0:N_train],     shape=(N_train, Nx+1, Ny+1), compression='gzip', chunks=True)
    print ("          Ouput Training data written!")
    

    dset_test_in   = hf.create_dataset("test_in",   data = FCN[N_train::], shape=(N_test, Nx+1, Ny+1),  compression='gzip', chunks=True)
    print ("          Input Test data written!")
    
    dset_test_out  = hf.create_dataset("test_out",  data = U[N_train::],     shape=(N_test, Nx+1, Ny+1),  compression='gzip', chunks=True)
    print ("          Output Test data written!")                      

    #print ("     Done writing Dataset to file!!")
               
    print()
    endd = time.time()
    print("Training Data Generation Ending after %s seconds "%(endd - startt))
    print()
    
    return fileName        


if __name__ == '__main__':

    xmin, xmax = 0, 1 #-1, 1
    ymin, ymax = 0, 1 #-1, 1
    Nx, Ny = 512, 512 #16, 16
    N_train, N_test = 5, 10 #1024, 5000 #2, 6
    _ = gen_data(xmin, xmax, ymin, ymax, Nx, Ny, N_train, N_test)
     