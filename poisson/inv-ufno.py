"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utils import *

import time
from datetime import datetime
from readData import readtoArray
import os, sys
from colorMap import parula
import argparse

torch.manual_seed(0)
np.random.seed(0)

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        
        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels*2, output_channels)
        self.deconv0 = self.deconv(input_channels*2, output_channels)
    
        self.output_layer = self.output(input_channels*2, output_channels, 
                                         kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)


    def forward(self, x):
        out_conv1 = self.conv1(x) #[10, 32, 48, 48] --> [10, 32, 24, 24]
        #print('size of x',x.size())        
        #print('size of out_conv1',out_conv1.size())
        out_conv2 = self.conv2_1(self.conv2(out_conv1)) #[10, 32, 24, 24] --> ?[10, 32, 12, 12]? --> [10, 32, 12, 12]
        out_conv3 = self.conv3_1(self.conv3(out_conv2)) #[10, 32, 12, 12] --> ?[10, 32,  6,  6]? --> [10, 32,  6,  6]
        #print('size of out_conv3',out_conv3.size())
        out_deconv2 = self.deconv2(out_conv3) #[10, 32, 6, 6] --> [10, 32, 12, 12]
        #print('size of out_conv2',out_conv2.size())
        #print('size of out_deconv2',out_deconv2.size())
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)

        return out

    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_planes, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias = False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                         stride=stride, padding=(kernel_size - 1) // 2)


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 7#15 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv5 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.unet3 = U_net(self.width, self.width, 3, 0)
        self.unet4 = U_net(self.width, self.width, 3, 0)
        self.unet5 = U_net(self.width, self.width, 3, 0)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.w4 = nn.Conv2d(self.width, self.width, 1)
        self.w5 = nn.Conv2d(self.width, self.width, 1)

        

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        #print('size of x',x.size())  
        x = F.pad(x, [0,self.padding, 0,self.padding])

        
        #print('size of x',x.size())  
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)
        #print('size of x',x.size())
        x1 = self.conv3(x)
        x2 = self.w3(x)#.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet3(x) 
        x = x1 + x2 + x3
        x = F.relu(x)        
        
        x1 = self.conv4(x)
        x2 = self.w4(x)#.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet4(x)
        x = x1 + x2 + x3
        x = F.relu(x)        
        
        x1 = self.conv5(x)
        x2 = self.w5(x)#.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet5(x)
        x = x1 + x2 + x3
        x = F.relu(x)

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


print("torch version is ",torch.__version__)
ntrain = 1000
ntest = 5000


learning_rate = 0.001

step_size = 100
gamma = 0.5

modes = 12
width = 32

parser = argparse.ArgumentParser(description='parse batch_size, epochs and resolution')
parser.add_argument('--bs',  default=10, type = int, help='batch-size')
parser.add_argument('--ep', default=500, type = int, help='epochs')
parser.add_argument('--res', default=32, type = int, help='resolution')
parser.add_argument('--wd', default=1e-4, type = float, help='weight decay')

args = parser.parse_args()

batch_size = args.bs #100
epochs = args.ep #500
res = args.res + 1#32#sys.argv[1]
wd = args.wd
print("\nUsing batchsize = %s, epochs = %s, and resolution = %s\n"%(batch_size, epochs, res))
params = {}
params["xmin"] = 0
params["ymin"] = 0
params["xmax"] = 1
params["ymax"] = 1

################################################################
# load data and data normalization
################################################################
PATH = "../../../../../../localdata/Derick/stuart_data/Darcy_421/fUG_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
#Read Datasets
Y_train, X_train, Y_test, X_test = readtoArray(PATH, 1024, 5000, Nx = 512, Ny = 512)

print ("Converting dataset to numpy array.")
tt = time.time()
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test  = np.array(X_test )
Y_test  = np.array(Y_test )
print ("    Conversion completed after %.2f minutes"%((time.time()-tt)/60))

print ("Subsampling dataset to the required resolution.")
tt = time.time()
X_train = SubSample(X_train, res, res)
Y_train = SubSample(Y_train, res, res)
X_test  = SubSample(X_test , res, res)
Y_test  = SubSample(Y_test , res, res)
print ("    Subsampling completed after %.2f minutes"%((time.time()-tt)/60))

print ("Taking out the required train/test size.")
tt = time.time()
x_train = torch.from_numpy(X_train[:ntrain, :, :]).float()
y_train = torch.from_numpy(Y_train[:ntrain, :, :]).float()
x_test  = torch.from_numpy(X_test[ :ntest,  :, :]).float()
y_test  = torch.from_numpy(Y_test[ :ntest,  :, :]).float()
print ("    Taking completed after %s seconds"%(time.time()-tt))
print("...")


x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)


x_train = x_train.reshape(ntrain,res,res,1)
x_test = x_test.reshape(ntest,res,res,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################
model = FNO2d(modes, modes, width).cuda()
print("Model has %s parameters"%(count_params(model)))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()
TIMESTAMP = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')
if os.path.isfile('files/inv/lossTrainData_'+TIMESTAMP+'.txt'):
    os.remove('files/inv/lossTrainData_'+TIMESTAMP+'.txt')
if os.path.isfile('files/inv/lossTestData_'+TIMESTAMP+'.txt'):
    os.remove('files/inv/lossTestData_'+TIMESTAMP+'.txt')   
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0


    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, res, res)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    abs_err = 0.0
    with torch.no_grad():

        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x).reshape(batch_size, res, res)
            out = y_normalizer.decode(out)

            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

    train_l2/= ntrain
    abs_err /= ntest
    test_l2 /= ntest

    t2 = default_timer()
    print("epoch: %s, completed in %.4f seconds. Training Loss: %.4f and Test Loss: %.4f"%(ep+1, t2-t1, train_l2, test_l2))

    file = open('files/inv/lossTrainData_'+TIMESTAMP+'.txt',"a")
    file.write(str(ep)+" "+str(train_l2)+"\n")#file.write(str(ep)+" "+str(train_mse)+"\n")
    file = open('files/inv/lossTestData_'+TIMESTAMP+'.txt',"a")
    file.write(str(ep)+" "+str(test_l2)+"\n")#file.write(str(ep)+" "+str(rel_err)+"\n")

ModelInfos = "_inv_%03d"%(res)+"~res_"+str(np.round(test_l2,6))+"~RelL2TestError_"+str(ntrain)+"~ntrain_"+str(ntest)+"~ntest_"+str(batch_size)+"~BatchSize_"+str(learning_rate)+\
            "~LR_"+str(wd)+"~Reg_"+str(gamma)+"~gamma_"+str(step_size)+"~Step_"+str(epochs)+"~epochs_"+time.strftime("%Y%m%d-%H%M%S")  
                            
torch.save(model.state_dict(), "files/inv/last_model"+ModelInfos+".pt")
os.rename('files/inv/lossTrainData_'+TIMESTAMP+'.txt', 'files/inv/lossTrainData'+ModelInfos+'.txt')
os.rename('files/inv/lossTestData_'+TIMESTAMP+'.txt', 'files/inv/lossTestData'+ModelInfos+'.txt')

dataTrain = np.loadtxt('files/inv/lossTrainData'+ModelInfos+'.txt')
dataTest  = np.loadtxt('files/inv/lossTestData'+ModelInfos+'.txt')
stepTrain = dataTrain[:,0] #Reading Epoch
stepTest  = dataTest[:,0]  #Reading Epoch                   
errorTrain = dataTrain[:,1] #Reading erros
errorTest  = dataTest[:,1]

print("Ploting Loss VS training step...")
fig = plt.figure(figsize=(15, 10))
plt.yscale('log')
plt.plot(stepTrain, errorTrain, label = 'Training Loss')
plt.plot(stepTest , errorTest , label = 'Test Loss')
plt.xlabel('epochs')#, fontsize=16, labelpad=15)
plt.ylabel('Loss')
plt.legend(loc = 'upper right')
plt.title("lr = %s test error = %s"%(learning_rate, str(np.round(test_l2,6))))
plt.savefig("figures/inv/Error_VS_TrainingStep"+ModelInfos+".png", dpi=500)


#def use_model():#(params, model,device,nSample,params):

model.load_state_dict(torch.load("files/inv/last_model"+ModelInfos+".pt"))
model.eval()


print()
print()

#Just a file containing data sampled in same way as the training and test dataset
fileName = "fUG_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
U_train, F_train, U_test, F_test= readtoArray(fileName, 1, 1, 512, 512)

F_train = SubSample(np.array(F_train), res, res)

print("Starting the Verification with Sampled Example")
tt = time.time()
U_FDM = SubSample(np.array(U_train), res, res)[0]

print("      Doing FNO on Example...")
tt = time.time()
ff = torch.from_numpy(F_train).float()
ff = x_normalizer.encode(ff)
ff = ff.reshape(1,res,res,1).cuda()#torch.cat([ff.reshape(1,res,res,1), grid.repeat(1,1,1,1)], dim=3).cuda()

U_FNO = y_normalizer.decode(model(ff).reshape(1, res, res)).detach().cpu().numpy()
U_FNO = U_FNO[0] 
print("            FNO completed after %s"%(time.time()-tt))

myLoss = LpLoss(size_average=False)
print()
print("Ploting comparism of FDM and FNO Simulation results")
fig = plt.figure(figsize=((5+2)*4, 5))

fig.suptitle("Plot of $- \Delta u = f(x, y)$ on $\Omega = ]0,1[ x ]0,1[$ with $u|_{\partial \Omega}  = 0.$")

colourMap = parula() #plt.cm.jet #plt.cm.coolwarm

plt.subplot(1, 4, 1)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("Input")
plt.imshow(F_train[0], cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 4, 2)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("FDM")
plt.imshow(U_FDM, cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 4, 3)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("UFNO")
plt.imshow(U_FNO, cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 4, 4)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("FDM-UFNO, RelL2Err = "+str(round(myLoss.rel_single(U_FNO, U_FDM).item(), 3)))
plt.imshow(np.abs(U_FDM - U_FNO), cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.savefig('figures/inv/compare'+ModelInfos+'.png',dpi=500)

plt.show()