import numpy as np
import torch

from utils import *

import time
from readData import readtoArray

torch.manual_seed(0)
np.random.seed(0)


dataPATHpoisson  = "../../../../../../localdata/Derick/stuart_data/Darcy_421/fUG_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
dataPathDarcyPWC = "../../../../../../localdata/Derick/stuart_data/Darcy_421/new_aUP_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
dataPathDarcyLN  = "../../../../../../localdata/Derick/stuart_data/Darcy_421/aUL_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
listdataPATH     = [dataPathDarcyPWC, dataPATHpoisson, dataPathDarcyLN]

normPATHpoisson  = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/poisson/UnitGaussianNormalizer/'
normPATHDarcyPWC = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/darcyPWC/UnitGaussianNormalizer/'
normPATHDarcyLN  = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/darcyLN/UnitGaussianNormalizer/'
listNORMPATH     = [normPATHDarcyPWC, normPATHpoisson, normPATHDarcyLN]

ntrain = 1000
ntest = 5000

for dataPATH, normPATH in zip(listdataPATH , listNORMPATH):

    X_train, Y_train, X_test, Y_test = readtoArray(dataPATH, 1024, 5000, Nx = 512, Ny = 512)

    print ("Converting dataset to numpy array.")
    tt = time.time()
    X_train0 = np.array(X_train)
    Y_train0 = np.array(Y_train)
    print ("    Conversion completed after %s seconds"%(time.time()-tt))

    for res in [32, 64, 128, 256, 512]:
        res = res + 1
        print ("Subsampling dataset to the required resolution.", res)
        tt = time.time()
        X_train = SubSample(X_train0, res, res)
        Y_train = SubSample(Y_train0, res, res)

        
        # old_res = res
        # res = closest_power(res)
        # X_train = CubicSpline3D(X_train, res, res)
        # Y_train = CubicSpline3D(Y_train, res, res)
        print ("    Subsampling completed after %s seconds"%(time.time()-tt))

        print ("Taking out the required train/test size.")
        tt = time.time()
        x_train = torch.from_numpy(X_train[:ntrain, :, :]).float()
        y_train = torch.from_numpy(Y_train[:ntrain, :, :]).float()
        print ("    Taking completed after %s seconds"%(time.time()-tt))
        print("...")


        x_normalizer = UnitGaussianNormalizer(x_train)

        y_normalizer = UnitGaussianNormalizer(y_train)

        torch.save(x_normalizer, normPATH+"param_normalizer-%s-res-%s-ntrain.pt"%(res-1, ntrain))
        torch.save(y_normalizer, normPATH+"solut_normalizer-%s-res-%s-ntrain.pt"%(res-1, ntrain))

        # torch.save(x_normalizer, normPATH+"param_normalizer-cs%s-res-%s-ntrain.pt"%(res-1, ntrain))
        # torch.save(y_normalizer, normPATH+"solut_normalizer-cs%s-res-%s-ntrain.pt"%(res-1, ntrain))
        
        #torch.save(x_normalizer, "normalisers/inv-%s-y_normalizer.pt"%(res-1))
        #torch.save(y_normalizer, "normalisers/inv-%s-x_normalizer.pt"%(res-1))