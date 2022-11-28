import numpy as np
#from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt
from scipy.special import softmax

from Functions import *

def deltax(mode="big"):
    if mode == "sobel":
        return np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
        
    if mode == "reduced":
        dxmask = np.array([[1/2, 0, -1/2]])
        return convolve2d(np.array([[0, 1, 0]]).T, dxmask)

    dxmask = np.array([[0, 1/2, 0, -1/2, 0]])
    return convolve2d(np.array([[0, 0, 1, 0, 0]]).T, dxmask)

def deltay(mode="big"):
    if mode =="sobel":
        return np.array([[1,   2,   1],
                         [0,   0,   0],
                         [-1, -2,  -1]])

    if mode == "reduced":
        dymask = np.array([[1/2, 0, -1/2]]).T
        return convolve2d(np.array([[0, 1, 0]]), dymask)

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

def Lv(inpic, shape = 'same', mode="reduced"):
    Lx = convolve2d(inpic, deltax(mode), shape)
    Ly = convolve2d(inpic, deltay(mode), shape)
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


def extractedge(inpic, scale, threshold, shape='same'):

    inpic_gauss = discgaussfft(inpic, sigma2=scale)
    zeropic = Lvvtilde(inpic_gauss, shape=shape)
    maskpic1 = Lvvvtilde(inpic_gauss, shape=shape) < 0

    curves = zerocrosscurves(zeropic, maskpic1)

    maskpic2 = Lv(inpic_gauss, shape=shape) > threshold
    curves = thresholdcurves(curves, maskpic2)

    return curves

def houghline(curves, magnitude, nrho, ntheta,
    threshold, nlines = 20, verbose = 1, soft=False):

    # Allocate accumulator space
    acc = np.zeros((nrho, ntheta))

    # Define a coordinate system in the accumulator space
    length_x, length_y =  magnitude.shape[0] // 2, magnitude.shape[1] // 2
    r = np.sqrt(length_x**2 + length_y**2)
    rho_vals = np.linspace(-r, r, nrho)
    theta_vals = np.linspace(-np.pi/2, np.pi/2, ntheta)
    soft_mag = np.sqrt(magnitude)/np.max(np.sqrt(magnitude))
        
    # Loop over all the edge points
    for edgepoint in zip(curves[0], curves[1]):
        # Check if valid point with respect to threshold
        x, y = edgepoint
        
        # Optionally, keep value from magnitude image
        curve_magnitude = magnitude[x,y]
        if curve_magnitude < threshold:
           continue

        # Loop over a set of theta values
        for j, theta in enumerate(theta_vals):

            # Compute rho for each theta value
            rho = (x - length_x) * np.cos(theta) + (y - length_y) * np.sin(theta)

            # Compute index values in the accumulator space
            i = np.argmin(abs(rho_vals - rho))

            # Update the accumulator
            acc[i,j] += 1 if not soft else soft_mag[x, y]


    # Extract local maxima from the accumulator
    pos, value, _ = locmax8(acc)

    # Delimit the number of responses if necessary
    indexvector = np.argsort(value)[-nlines:]
    pos = pos[indexvector]

    linepar = []

    if verbose==2:
        fig = plt.figure()
        # Overlay these curves on the gradient magnitude image
        showgrey(magnitude, False)


    for idx in range(nlines):
        thetaidxacc = pos[idx, 0]
        rhoidxacc = pos[idx, 1]

        max_theta = theta_vals[thetaidxacc]
        max_rho = rho_vals[rhoidxacc]

        linepar.append([max_theta, max_rho])
        

        if verbose==2:
            # Compute a line for each one of the strongest responses in the accumulator
            x0 = max_rho * np.cos(max_theta) + length_x
            y0 = max_rho * np.sin(max_theta) + length_y

            if abs(np.sin(max_theta)) < 1e3:
                dx = length_x*(-np.cos(max_theta)/np.sin(max_theta))
            else: dx = 0

            if abs(np.cos(max_theta)) < 1e3:
                dy = length_y*(-np.sin(max_theta)/np.cos(max_theta))
            else: dy = 0

            dx = r * (-np.sin(max_theta))
            dy = r * (np.cos(max_theta))

            plt.plot([y0-dy, y0, y0+dy], [x0-dx, x0, x0+dx], 'r-')

    if verbose==3:
        plt.show()
        fig = plt.figure()
        plt.imshow(acc)
        plt.ylabel(r"$\rho$")
        plt.xlabel(r"$\theta$")
        plt.xticks(np.linspace(0, len(theta_vals)-1, 5),
                   np.round(np.linspace(-np.pi/2, np.pi/2, 5), 2))
        plt.yticks(np.linspace(0, len(rho_vals)-1, 5),
                   np.round(np.linspace(-r, r, 5), 2))
        plt.show()

    # Return the output data [linepar, acc]
    return linepar, acc

def houghedgeline(pic, scale, gradmagnthreshold, nrho,
                  ntheta, nlines = 20, verbose = 1, soft=False):
    
    curves = extractedge(pic, scale, gradmagnthreshold, "same")
    gaussianSmooth = discgaussfft(pic, scale)
    gradmagn = Lv(gaussianSmooth, "same")

    linepar, acc = houghline(curves, gradmagn, nrho,
    ntheta, gradmagnthreshold,
    nlines, verbose, soft)
    
    if verbose==1:
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1,3,1)
        overlaycurves(pic, curves)
        ax.set_title("Original w/ edges")
        length_x, length_y = gradmagn.shape[0]//2, gradmagn.shape[1]//2
        r = np.sqrt(length_x**2 + length_y**2)
        
        ax = fig.add_subplot(1,3,2)
        showgrey(pic, False)
        
        for (max_theta, max_rho) in linepar:
            
            # Compute a line for each one of the strongest responses in the accumulator
            x0 = max_rho * np.cos(max_theta) + length_x
            y0 = max_rho * np.sin(max_theta) + length_y

            if abs(np.sin(max_theta)) < 1e3:
                dx = length_x*(-np.cos(max_theta)/np.sin(max_theta))
            else: dx = 0

            if abs(np.cos(max_theta)) < 1e3:
                dy = length_y*(-np.sin(max_theta)/np.cos(max_theta))
            else: dy = 0

            dx = r * (-np.sin(max_theta))
            dy = r * (np.cos(max_theta))

            plt.plot([y0-dy, y0, y0+dy], [x0-dx, x0, x0+dx], 'r-')

        ax.set_title("Lines over original")
        
        ax = fig.add_subplot(1,3,3)
        plt.imshow(acc)
        ax.set_ylabel(r"$\rho$")
        ax.set_xlabel(r"$\theta$")
        ax.set_xticks(np.linspace(0, acc.shape[1]-1, 5),
                    np.round(np.linspace(-np.pi/2, np.pi/2, 5), 2))
        
        ax.set_yticks(np.linspace(0, acc.shape[0]-1, 5),
                    np.round(np.linspace(-r, r, 5), 2))
        ax.set_title("Parameter space")
        
        title = f"scale={scale}, threshold={gradmagnthreshold}, nrho={nrho}, ntheta={ntheta}"
        if soft:
            title += ". With soft voting"
            
        fig.suptitle(title)
        plt.show()
        
    return linepar, acc, gradmagn


def ex1():
    tools = np.load("Images-npy/few256.npy")
    dxtools = convolve2d(tools, deltax(mode="sobel"), 'valid')
    dytools = convolve2d(tools, deltay(mode="sobel"), 'valid')

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


def ex2(img, thresholds, show_hist=False, sigma=None):
    imgs = {"tools":np.load("Images-npy/few256.npy"),
            "house":np.load("Images-npy/godthem256.npy")
            }

    if sigma is not None:
        inpic = discgaussfft(imgs[img], sigma2=sigma)
    else:
        inpic = imgs[img]

    gradmagn = Lv(inpic, mode="sobel")
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    showgrey(imgs[img], False)

    ax1.set_title("Original")

    ax2 = fig.add_subplot(1, 2, 2)
    showgrey(gradmagn, False)

    if sigma is not None:
        ax2.set_title(f"Gradient magnitude, sigma={sigma}")
    else:
        ax2.set_title("Gradient magnitude")

    plt.show()

    if show_hist:
        plt.figure(figsize=(10, 5))
        plt.hist(gradmagn)
        plt.xticks(np.arange(0, int(np.max(gradmagn)), step=50))

        plt.show()

    fig = plt.figure()

    for i, threshold in enumerate(thresholds):

        ax = fig.add_subplot(1, len(thresholds), i+1)
        showgrey((gradmagn > threshold).astype(int), False)
        ax.set_title(f"Threshold={threshold}")

    plt.show()


def ex3():
    x, y = np.meshgrid(range(-5, 6), range(-5, 6))

    print(convolve2d(x**3, DELTA_XXX, "valid"))
    print(convolve2d(x**3, DELTA_XX, "valid"))
    print(6*x)
    print(convolve2d(x**2*y, DELTA_XXY, "valid"))

    house = np.load("Images-npy/godthem256.npy")
    scales = [0.0001, 1.0, 4.0, 16.0, 64.0]

    fig = plt.figure()
    for i, scale in enumerate(scales):
        ax = fig.add_subplot(1, len(scales), i+1)
        showgrey(contour(Lvvtilde(discgaussfft(house, scale), 'same')), False)
        ax.set_title(f"scale={scale}")

    plt.show()

    tools = np.load("Images-npy/few256.npy")

    fig = plt.figure()
    for i, scale in enumerate(scales):
        ax = fig.add_subplot(1, len(scales), i+1)
        showgrey((Lvvvtilde(discgaussfft(tools, scale), 'same')<0).astype(int), False)
        ax.set_title(f"scale={scale}")

    plt.show()


def ex4():

    imgs = [np.load("Images-npy/few256.npy"),
            np.load("Images-npy/godthem256.npy")]

    scales = [1.0, 1.5, 2.0, 4.0, 8.0]
    thresholds = [2, 5, 8, 10]

    # selection of scale and threshold
    for img in imgs:

        fig = plt.figure()
        for i, threshold in enumerate(thresholds):
            for j, scale in enumerate(scales):
                ax = fig.add_subplot(len(thresholds), len(scales), i*len(scales) + j +1)
                edgecurves = extractedge(img, scale=scale, threshold=threshold)
                overlaycurves(img, edgecurves)
                ax.set_title(f"t={threshold}, scale={scale}")

    plt.show()

    # best results
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    edgecurves = extractedge(imgs[0], scale=4, threshold=5)
    overlaycurves(imgs[0], edgecurves)
    ax.set_title(f"t={5}, scale={4}")

    ax = fig.add_subplot(1, 2, 2)
    edgecurves = extractedge(imgs[1], scale=2, threshold=5)
    overlaycurves(imgs[1], edgecurves)
    ax.set_title(f"t={5}, scale={2}")


    plt.show()

    
def ex5(img, scale, gradmagnthreshold, nrho, ntheta, nlines, soft=False):
    testimage1 = np.load("Images-npy/triangle128.npy")
    smalltest1 = binsubsample(testimage1)
    
    testimage2 = np.load("Images-npy/houghtest256.npy")
    smalltest2 = binsubsample(binsubsample(testimage2))
    
    imgs = {
        "triangle": smalltest1,
        "shapes": smalltest2,
        "tools": np.load("Images-npy/few256.npy"),
        "calc": np.load("Images-npy/phonecalc256.npy"),
        "house": np.load("Images-npy/godthem256.npy")
    }
        
    linepar, acc, gradmagn = houghedgeline(imgs[img],
                                 scale=scale,
                                 gradmagnthreshold=gradmagnthreshold,
                                 nrho=nrho,
                                 ntheta=ntheta,
                                 nlines=nlines,
                                 soft=soft,
                                 verbose=1)


if __name__ == '__main__':

    exercise = 6

    if exercise==1:
        # Q1
        ex1()
    elif exercise==2:
        # Q2, Q3
        ex2("tools", [50, 100, 200, 300, 500], True, None)
        ex2("tools", [50, 100, 120, 200], True, 4)
        ex2("house", [100, 200, 300, 400, 500], True, None)
        ex2("house", [50, 100, 200, 300], True, 2)
        ex2("house", [50, 100, 200, 300], False, 4)
    elif exercise==3:
        # Q4, Q5, Q6
        ex3()
    elif exercise==4:
        # Q7, 
        ex4()
    elif exercise==5:
        # Q8, Q9
        img = "house"
        
        if img=="traingle":
            ex5(img, scale=2, gradmagnthreshold=20,
                nrho=100, ntheta=100, nlines=3)
            
        elif img == "shapes":
            ex5(img, scale=2, gradmagnthreshold=20,
                nrho=50, ntheta=50, nlines=10)
            
            ex5(img, scale=2, gradmagnthreshold=20,
                nrho=200, ntheta=200, nlines=10)
            
            ex5(img, scale=2, gradmagnthreshold=20,
                nrho=200, ntheta=100, nlines=10)
            
            ex5(img, scale=2, gradmagnthreshold=20,
                nrho=400, ntheta=400, nlines=10)
            
        elif img == "tools":
            ex5(img, scale=4, gradmagnthreshold=5,
                nrho=200, ntheta=200, nlines=20)
            
        elif img == "calc":
            ex5(img, scale=10, gradmagnthreshold=2,
                nrho=300, ntheta=150, nlines=25)
            
        elif img == "house":
            ex5(img, scale=2, gradmagnthreshold=5,
                nrho=150, ntheta=100, nlines=20)
            
    elif exercise==6:
        # Q10
        ex5("tools", scale=4, gradmagnthreshold=5,
                nrho=200, ntheta=200, nlines=20, soft=True)
        
        ex5("house", scale=2, gradmagnthreshold=5,
                nrho=250, ntheta=200, nlines=30, soft=True)
        
        ex5("calc", scale=10, gradmagnthreshold=2,
                nrho=300, ntheta=150, nlines=25, soft=True)
        
