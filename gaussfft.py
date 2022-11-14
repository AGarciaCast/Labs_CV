import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from Functions import showfs, showgrey
from matplotlib import pyplot as plt

def gaussfft(pic, t, plot_kernel=False):

    h, w = np.shape(pic)
    x, y = np.meshgrid(np.linspace(-h//2, h//2, h), np.linspace(-w//2, w//2, w))
    kernel = 1/(2*np.pi*t) * np.exp(-(x**2 + y**2)/(2*t))
    
    if plot_kernel:
        showgrey(kernel, False)
        plt.title(r"$Kernel, t={}$".format(t))
        plt.show()
    
    kfft = fft2(kernel)
    pfft = fft2(pic)
        
    result = np.real(fftshift(ifft2(kfft * pfft)))

    return result
