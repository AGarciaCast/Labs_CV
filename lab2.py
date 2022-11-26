import numpy as np
#from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt

from Functions import *
from gaussfft import gaussfft


def deltax(sobel=False):
    if sobel:
        return np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    
    dxmask = np.array([[0, 1/2, 0, -1/2, 0]])
    return convolve2d(np.array([[0, 0, 1, 0, 0]]).T, dxmask)

def deltay(sobel=False):
    if sobel:
        return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
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

def Lv(inpic, shape = 'same', sigma=None, sobel=False):
    
    if sigma is not None:
        inpic = gaussfft(inpic, sigma)
    
    Lx = convolve2d(inpic, deltax(sobel), shape)
    Ly = convolve2d(inpic, deltax(sobel), shape)
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

			
def ex1():
	tools = np.load("Images-npy/few256.npy")
	dxtools = convolve2d(tools, deltax(), 'valid')
	dytools = convolve2d(tools, deltay(), 'valid')
	
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 3, 1)
	showgrey(tools, False)
	ax1.set_title(f"Original, size={tools.shape}")
	ax2 = fig.add_subplot(1, 3, 2)
	showgrey(dxtools, False)
	ax2.set_title(f"dx, size={dxtools.shape}")
	ax3 = fig.add_subplot(1, 3, 3)
	showgrey(dytools, False)
	ax3.set_title(f"dy, size={dytools.shape}")

	plt.show()


def ex2(show_hist=False, sigma=None):
	tools = np.load("Images-npy/godthem256.npy")
	gradmagntools = Lv(tools, sigma=sigma, sobel=True)
 
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 2, 1)
	showgrey(tools, False)

	ax1.set_title("Original")
 
	ax2 = fig.add_subplot(1, 2, 2)
	showgrey(gradmagntools, False)
 
	if sigma is not None: 
		ax2.set_title(f"Gradient magnitude, sigma={sigma}")
	else:
		ax2.set_title("Gradient magnitude")
	plt.show()
 
	if show_hist:
		plt.figure(figsize=(10, 5))
		plt.hist(gradmagntools) 
		plt.xticks(np.arange(0, int(np.max(gradmagntools)), step=50))
	
		plt.show()
  
	thresholds = [50, 100, 200, 300]

	fig = plt.figure()

	for i, threshold in enumerate(thresholds):

		ax = fig.add_subplot(1, len(thresholds), i+1)
		showgrey((gradmagntools > threshold).astype(int), False)
		ax.set_title(f"Threshold={threshold}")

	plt.show()
 

def ex3():
    x, y = np.meshgrid(range(-5, 6), range(-5, 6))
    
    print(convolve2d(x**3, DELTA_XXX, "valid"))
    print(convolve2d(x**3, DELTA_XX, "valid"))
    print(6*x)
    print(convolve2d(x**2*y, DELTA_XXY, "valid"))
   
if __name__=='__main__':
    
    exercise = 3 
    if exercise == 1:
        ex1()
    elif exercise == 2:
        ex2(True, 4)
    elif exercise == 3:
        ex3()
        
