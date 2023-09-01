import numpy as np
import torch

from utils import *

import time
from readData import readtoArray

torch.manual_seed(0)
np.random.seed(0)

PATH = "../../../../../../localdata/Derick/stuart_data/Darcy_421/fUG_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
ntrain = 1000
ntest = 5000

X_train, Y_train, X_test, Y_test = readtoArray(PATH, 1024, 5000, Nx = 512, Ny = 512)

print ("Converting dataset to numpy array.")
tt = time.time()
X_train0 = np.array(X_train)
Y_train0 = np.array(Y_train)
X_test0  = np.array(X_test )
Y_test0  = np.array(Y_test )
print ("    Conversion completed after %s seconds"%(time.time()-tt))

for res in [32, 64, 128, 256, 512]:
    res = res + 1
    print ("Subsampling dataset to the required resolution.", res)
    tt = time.time()
    X_train = SubSample(X_train0, res, res)
    Y_train = SubSample(Y_train0, res, res)
    X_test  = SubSample(X_test0 , res, res)
    Y_test  = SubSample(Y_test0 , res, res)
    print ("    Subsampling completed after %s seconds"%(time.time()-tt))

    print ("Taking out the required train/test size.")
    tt = time.time()
    x_train = torch.from_numpy(X_train[:ntrain, :, :]).float()
    y_train = torch.from_numpy(Y_train[:ntrain, :, :]).float()
    x_test  = torch.from_numpy(X_test[ :ntest,  :, :]).float()
    y_test  = torch.from_numpy(Y_test[ :ntest,  :, :]).float()
    print ("    Taking completed after %s seconds"%(time.time()-tt))
    print("...")


    x_normalizer = UnitGaussianNormalizer(x_train)

    y_normalizer = UnitGaussianNormalizer(y_train)

    torch.save(x_normalizer, "files/fwd-%s-x_normalizer.pt"%(res-1))
    torch.save(y_normalizer, "files/fwd-%s-y_normalizer.pt"%(res-1))   
    
    torch.save(x_normalizer, "files/inv-%s-y_normalizer.pt"%(res-1))
    torch.save(y_normalizer, "files/inv-%s-x_normalizer.pt"%(res-1))