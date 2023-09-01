import torch
import numpy as np
import matplotlib.pyplot as plt
import time
#import torch.fft as tfft
import math
from colorMap import parula

writestep = 100

def sample_f(batch_size=1, n=64, m=64):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    #torch.cuda.manual_seed_all(9)
    #torch.manual_seed(10)
    f = torch.randn(batch_size, n, m)#.to(device)
    f = f-torch.mean(f,(1,2), True)

    t = time.time()

    
    #t = time.time()
    #den1 = np.zeros((n,m),dtype='complex_')
    #for p in range(n):
    #    for q in range(m):
    #        den1[p,q] = (9 + (-1j*2*math.pi*p)**2 + (-1j*2*math.pi*q)**2)**2
    #print("Done computing after %s"%(time.time()-t))   
    #u1 = tfft.ifft2(fhat/den1)
                                  
    return torch.real(f)#.detach().cpu() #, torch.real(u1)
    
    
if __name__ == '__main__':
    Nx = 512
    Ny = 512
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1

    dx = float(xmax-xmin)/float(Nx)
    dy = float(ymax-ymin)/float(Ny)
    

    f = sample_f(batch_size=6000, n=Nx+1, m=Ny+1) 

    uG = f[0]#-torch.mean(f[0])
    uL = torch.exp(uG)
    uP = torch.zeros_like(uG)
    uP[uG > 0] = 12
    uP[uG < 0] = 3

    colourMap = parula()#plt.cm.jet #plt.cm.coolwarm
    
    fig = plt.figure(figsize=(18, 5))

    fig.suptitle("Sample")
    plt.subplot(1, 3, 1)

    plt.xlabel('x')#, fontsize=16, labelpad=15)
    plt.ylabel('y')#, fontsize=16, labelpad=15)
    plt.title("$\mu_G$")
    plt.imshow(uG, cmap=colourMap, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
    plt.colorbar()

    plt.subplot(1, 3, 2)

    plt.xlabel('x')#, fontsize=16, labelpad=15)
    plt.ylabel('y')#, fontsize=16, labelpad=15)
    plt.title("$\mu_L$")
    plt.imshow(uL, cmap=colourMap, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
    plt.colorbar()

    plt.subplot(1, 3, 3)

    plt.xlabel('x')#, fontsize=16, labelpad=15)
    plt.ylabel('y')#, fontsize=16, labelpad=15)
    plt.title("$\mu_P$")
    plt.imshow(uP, cmap=colourMap, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
    plt.colorbar()        

    t = time.localtime()    
    plt.savefig("figures/sample"+time.strftime('%Y%m%d-%H:%M:%S', t)+"jpg")
    
    plt.show()