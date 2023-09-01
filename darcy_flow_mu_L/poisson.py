import sys
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

def poisson(xmin, xmax, ymin, ymax, Nx, Ny, \
            gNorth, gWest, gSouth, gEast, fcn):
    """
    numerical solution of Poisson equation on a rectangle with
    Dirichlet boundary condition at all boundaries. The boundary
    conditions are defined via the boundary functions gNorth(x),
    gWest, gSouth, and gEast, respectively
    we apply direct solutions of the resulting liner system of eqations.
    """
    #print('Poisson equation with Dirichlet conditions on all boundaries') 
    #print('Method: poisson, direct solution')

    dx = float(xmax-xmin)/float(Nx)
    dy = float(ymax-ymin)/float(Ny)
    beta = dx/dy
    delta = dy/dx
    #print ('dx = dy = {}'.format(dx))
     
    x = np.linspace(xmin,xmax,Nx+1)
    y = np.linspace(ymin,ymax,Ny+1)

    X,Y = np.meshgrid(x,y)

    # generate 1D Laplace operator in x direction
    datx = -np.ones((3,Nx-1))
    datx[1,:] = 2.0     # diagonal entries
    Dxx = sps.spdiags(datx,[-1,0,1],Nx-1,Nx-1)
    Ix = sps.eye(Nx-1)

    # generate 1D Laplace operator in y direction
    daty = -np.ones((3,Ny-1))
    daty[1,:] = 2.0     # diagnal entries
    Dyy = sps.spdiags(daty,[-1,0,1],Ny-1,Ny-1)
    Iy = sps.eye(Ny-1)

    # generate 2D Laplace operator in x and y direction
    #A = sps.kronsum(Dxx,Dyy)
    A = delta*sps.kron(Iy,Dxx) + beta*sps.kron(Dyy,Ix)
    #print ('number of elements is',A.count_nonzero())# np.count_nonzero(A.toarray()))

    # Make gNorth, gWest, gSouth and gEast functions 
    # if they are float/int
    if isinstance(gNorth, (float,int)):
        _gNorth = float(gNorth)  # Make copy of gNorth
        gNorth = lambda x: _gNorth
    if isinstance(gWest, (float,int)):
        _gWest = float(gWest)  # Make copy of gWest
        gWest = lambda y: _gWest
    if isinstance(gSouth, (float,int)):
        _gSouth = float(gSouth)  # Make copy of gSouth
        gSouth = lambda x: _gSouth
    if isinstance(gEast, (float,int)):
        _gEast = float(gEast)  # Make copy of gEast
        gEast = lambda y: _gEast

    # define the array for the discrete solution and
    # set Dirichlet boundary conditions.
    u = np.zeros((Ny+1,Nx+1))
    u[:,0]  = gWest(y)    # west boundary
    u[:,-1] = gEast(y)    # east boundary
    u[0,:]  = gSouth(x)   # south boundary
    u[-1,:] = gNorth(x)   # north boundary

    # fix the corner values of the u-matrix
    u[0,0]   = 0.5*(gSouth(xmin) + gWest(ymin))  # south west corner
    u[-1,0]  = 0.5*(gNorth(xmin) + gWest(ymax))  # north west corner 
    u[-1,-1] = 0.5*(gNorth(xmax) + gEast(ymax))  # north east corner
    u[0,-1]  = 0.5*(gSouth(xmax) + gEast(ymin))  # south east corner 


    # evaluate the inhomogeneity
    rhs = np.zeros((Ny-1,Nx-1))
    if fcn is None:
        rhs[:,:] = np.zeros((Ny-1,Nx-1))
    elif isinstance(fcn, (float,int)):
        rhs[:,:] = float(fcn)*np.ones((Ny-1,Nx-1))
    elif isinstance(fcn, np.ndarray):
        if np.shape(fcn) != np.shape(X):
            print ("Check the dimensions of the rhs")
        else:
            rhs[:,:] = dx*dy*fcn[1:-1,1:-1]      
    else:
        rhs[:,:] = dx*dy*fcn(X[1:-1,1:-1],Y[1:-1,1:-1])
        
    # eliminate Dirichlet conditions from the linear system
    rhs[:,0]  += delta*u[1:-1,0]    # west boundary
    rhs[:,-1] += delta*u[1:-1,-1]   # east boundary
    rhs[0,:]  +=  beta*u[-1,1:-1]#beta*u[0,1:-1]    # north boundary
    rhs[-1,:] +=  beta*u[0,1:-1]#beta*u[-1,1:-1]   # south boundary

    # linearize right hand side vector and solve the system
    b = rhs.flatten()
    
    x = spsolve(A,b)
    u[1:-1,1:-1] = x.reshape(Ny-1,Nx-1)

    # Output
    return u, X, Y

if __name__ == '__main__':
    Nx = 160
    Ny = 160

    xmin, xmax = 0., 1.
    ymin, ymax = 0., 1.

    def fun(x,y):
        # exact solution
        return 10.  # constant

    def gNorth(x):
        # north boundary condition
        return 0.   # y = 1

    def gSouth(x):
        # south boundary condition
        return 0.   # y = 0

    def gWest(y):
        # west boundary condition
        return 0.   # x = 0

    def gEast(y):
        # east boundary condition
        return 0.   # x = 1


    u,X,Y = poisson(xmin, xmax, ymin, ymax, Nx, Ny, \
                    gNorth, gWest, gSouth, gEast, fun)

