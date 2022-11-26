import numpy as np
#from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt

from Functions import *
from gaussfft import gaussfft


def deltax():
    dxmask = np.array([[0, 1/2, 0, -1/2, 0]])
    return convolve2d(np.array([[0, 0, 1, 0, 0]]).T, dxmask)

def deltay():
    dymask = np.array([[0, 1/2, 0, -1/2, 0]]).T
    return convolve2d(np.array([[0, 0, 1, 0, 0]]), dymask)

def deltaxx():
    dxmask = np.array([[0, 1, -2, 1, 0]])
    return convolve2d(np.array([[0, 0, 1, 0, 0]]).T, dxmask)

def deltayy():
    dymask = np.array([[0, 1, -2, 1, 0]]).T
    return convolve2d(np.array([[0, 0, 1, 0, 0]]), dymask)

DELTA_X = deltax()
DELTA_Y = deltay()
DELTA_XX = deltaxx()
DELTA_XY = convolve2d(DELTA_X, DELTA_Y, "same")
DELTA_YY = deltayy()
DELTA_XXX = convolve2d(DELTA_X, DELTA_XX, "same")
DELTA_XXY = convolve2d(DELTA_XX, DELTA_Y, "same")
DELTA_XYY = convolve2d(DELTA_X, DELTA_YY, "same")
DELTA_YYY = convolve2d(DELTA_Y, DELTA_YY, "same")

def Lv(inpic, shape = 'same'):
    Lx = convolve2d(inpic, DELTA_X, shape)
    Ly = convolve2d(inpic, DELTA_Y, shape)
    return np.sqrt(Lx**2 + Ly**2)

def Lvvtilde(inpic, shape = 'same'):
    Lx = convolve2d(inpic, DELTA_X, shape)
    Ly = convolve2d(inpic, DELTA_Y, shape)
    Lxx = convolve2d(inpic, DELTA_XX, shape)
    Lxy = convolve2d(inpic, DELTA_XY, shape)
    Lyy = convolve2d(inpic, DELTA_YY, shape)
    
    return (Lx**2)*Lxx + 2*Lx*Ly*Lxy + (Ly**2)*Lyy

def Lvvvtilde(inpic, shape = 'same'):
    Lx = convolve2d(inpic, DELTA_X, shape)
    Ly = convolve2d(inpic, DELTA_Y, shape)
    Lxxx = convolve2d(inpic, DELTA_XXX, shape)
    Lxxy = convolve2d(inpic, DELTA_XXY, shape)
    Lxyy = convolve2d(inpic, DELTA_XYY, shape)
    Lyyy = convolve2d(inpic, DELTA_YYY, shape)
    
    return (Lx**3)*Lxxx + 3*(Lx**2)*Ly*Lxxy + 3*Lx*(Ly**2)*Lxyy + (Ly**3)*Lyyy
	

def extractedge(inpic, scale, threshold, shape):
    return contours

def houghline(curves, magnitude, nrho, ntheta,
              threshold, nlines = 20, verbose = False):
    return linepar, acc

def houghedgeline(pic, scale, gradmagnthreshold, nrho,
                  ntheta, nlines = 20, verbose = False):
    return linepar, acc

			
if __name__=='__main__':
    x, y = np.meshgrid(range(-5, 6), range(-5, 6))
    
    print(convolve2d(x**3, DELTA_XXX, "valid"))
    print(convolve2d(x**3, DELTA_XX, "valid"))
    print(6*x)
    print(convolve2d(x**2*y, DELTA_XXY, "valid"))