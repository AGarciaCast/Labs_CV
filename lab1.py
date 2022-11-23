import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d

from Functions import *
from gaussfft import gaussfft
from fftwave import fftwave


# Either write your code in a file like this or use a Jupyter notebook.
#
# A good idea is to use switches, so that you can turn things on and off
# depending on what you are working on. It should be fairly easy for a TA
# to go through all parts of your code though.


def ex1():
    fftwave(5, 65)
    fftwave(9, 5)
    fftwave(17, 9)
    fftwave(17, 121)
    fftwave(5, 1)
    fftwave(125, 1)

def ex2(use_log=True):
    F = np.concatenate([np.zeros((56,128)), np.ones((16,128)), np.zeros((56,128))])
    G = F.T
    H = F + 2*G
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 3, 1)
    showgrey(F, False)
    ax1.set_title(r"$F$")
    ax2 = fig.add_subplot(3, 3, 2)
    showgrey(G, False)
    ax2.set_title(r"$G=F^T$")
    ax3 = fig.add_subplot(3, 3, 3)
    showgrey(H, False)
    ax3.set_title(r"$H=F + 2*G$")
    Fhat = fft2(F)
    Ghat = fft2(G)
    Hhat = fft2(H)

    ax4 = fig.add_subplot(3, 3, 4)
    showgrey(np.log(1 + np.abs(Fhat)), False)
    ax4.set_title(r"$\hat{F}$")

    ax5 = fig.add_subplot(3, 3, 5)
    showgrey(np.log(1 + np.abs(Ghat)), False)
    ax5.set_title(r"$\hat{G}}$")

    ax6 = fig.add_subplot(3, 3, 6)
    showgrey(np.log(1 + np.abs(Hhat)), False)
    ax6.set_title(r"$\hat{H}}$")

    ax7 = fig.add_subplot(3, 3, 7)
    shift_Fhat = fftshift(Fhat)
    if use_log:
        showgrey(np.log(1 + np.abs(shift_Fhat)), False)
    else:
        showgrey(np.abs(shift_Fhat), False)
    ax7.set_title(r"$\hat{F}_{centered}$")


    ax8 = fig.add_subplot(3, 3, 8)
    shift_Ghat = fftshift(Ghat)
    if use_log:
        showgrey(np.log(1 + np.abs(shift_Ghat)), False)
    else:
        showgrey(np.abs(shift_Ghat), False)
        
    ax8.set_title(r"$\hat{G}_{centered}$")

    ax9 = fig.add_subplot(3, 3, 9)
    shift_Hhat = fftshift(Hhat)
    if use_log:
        showgrey(np.log(1 + np.abs(shift_Hhat)), False)
        # showfs(Hhat, False)
    else:
        showgrey(np.abs(shift_Hhat), False)
        
    ax9.set_title(r"$\hat{H}_{centered}$")


    plt.show()

def ex3():
    F = np.concatenate([np.zeros((56, 128)), np.ones((16, 128)), np.zeros((56, 128))])
    G = F.T
    Fhat = fft2(F)
    Ghat = fft2(G)
    fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1)
    showgrey(F, False)
    ax.set_title(r"$F")
    ax2 = fig.add_subplot(2, 3, 2)
    showgrey(G, False)
    ax2.set_title(r"G")
    ax3 = fig.add_subplot(2, 3, 3)
    showgrey(F * G, False)
    ax3.set_title(r"$H = F*G$")
    ax4 = fig.add_subplot(2, 3, 4)
    showfs(fft2(F), False)
    ax4.set_title(r"$\hat{F}$")
    ax5 = fig.add_subplot(2, 3, 5)
    showfs(fft2(G), False)
    ax5.set_title(r"$\hat{G}$")
    ax6 = fig.add_subplot(2, 3, 6)
    showfs(fft2(F*G), False)
    ax6.set_title(r"$\hat{H}$")
    plt.show()
    
    
    # Stackoverflow - uncomment to get the image convolution result
    # showfs(fftshift(convolve2d(fftshift(Fhat)/128, fftshift(Ghat)/128, mode="same")))
    
    # using (fake) circular convolution
    
    Xf = np.fft.fft2(F)
    Yf = np.fft.fft2(G)
    
    # fourier of Y
    showfs(Yf)
    N = Xf.size    # or Yf.size since they must have the same size
    # make the signal periodic
    Zf = np.vstack([Yf, Yf, Yf])
    Zf = np.hstack([Zf, Zf, Zf])
    # show the periodized signal
    showfs(Zf)
    print(Zf.shape)
    # why we need inverse shift
    conv = np.fft.ifftshift(convolve2d(Xf, Zf, mode="same")/128**2)
    showfs(conv)



def ex4():
    F_ = np.concatenate([np.zeros((56, 128)), np.ones((16, 128)), np.zeros((56, 128))])
    # showgrey(F_)
    # showfs(fft2(F_))
    
    F = np.concatenate([np.zeros((60, 128)), np.ones((8, 128)), np.zeros((60, 128))]) * \
        np.concatenate([np.zeros((128, 48)), np.ones((128, 32)), np.zeros((128, 48))], axis=1)
    
    G = F.T
    Fhat = fft2(F)
    Ghat = fft2(G)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    showgrey(F, False)
    ax1.set_title(r"$F$")
    ax2 = fig.add_subplot(2, 3, 2)
    showgrey(G, False)
    ax2.set_title(r"$G$")
    ax3 = fig.add_subplot(2, 3, 3)
    showgrey(F * G, False)
    ax3.set_title(r"$F*G$")
    ax4 = fig.add_subplot(2, 3, 4)
    showfs(fft2(F), False)
    ax4.set_title(r"$\hat{F}$")
    ax5 = fig.add_subplot(2, 3, 5)
    showfs(fft2(G), False)
    ax5.set_title(r"$\hat{G}$")
    ax6 = fig.add_subplot(2, 3, 6)
    showfs(fft2(F*G), False)
    ax6.set_title(r"$\widehat{F*G}$")
    
    plt.show()


def ex5():
    
       
    F = np.concatenate([np.zeros((60, 128)), np.ones((8, 128)), np.zeros((60, 128))]) * \
        np.concatenate([np.zeros((128, 48)), np.ones((128, 32)), np.zeros((128, 48))], axis=1)
    
    G = F.T
       
    alpha = [30, 45, 60, 90]
    
    fig = plt.figure()
    for i, angle in enumerate(alpha):
        ax1 = fig.add_subplot(3, len(alpha), i+1)
        ax1.set_title(r"$G, \alpha={}^o$".format(angle))
        G = rot(F, angle)
        showgrey(G, False)
        
        ax2 = fig.add_subplot(3, len(alpha), len(alpha)+i+1)
        ax2.set_title(r"$\hat{G}$")
        Ghat = fft2(G)
        showfs(Ghat, False)
        
        ax3 = fig.add_subplot(3, len(alpha), 2*len(alpha)+i+1)
        ax3.set_title(r"$\hat{G}_{rot}, \alpha=0^o$")
        Hhat = rot(fftshift(Ghat), -angle)
        showgrey(np.log(1 + abs(Hhat)), False)
        
    plt.show()

def ex6():
    
    
    images = ['phonecalc128.npy', 'office128.npy']
    fig = plt.figure()
    for i, loc in enumerate(images):
        img = np.load(f"images-npy/{loc}")
        ax1 = fig.add_subplot(3, len(images), i+1)
        ax1.set_title("{}, original phase".format(loc.split(".")[0]))
        showgrey(img, False)
        
        ax2 = fig.add_subplot(3, len(images), len(images)+i+1)
        ax2.set_title("Different power spectrum")

        showgrey(pow2image(img), False)
        
        ax3 = fig.add_subplot(3, len(images), 2*len(images)+i+1)
        ax3.set_title("Random phase")

        showgrey(randphaseimage(img), False)
        
    plt.show()
    
def ex7(plot_kernel=True):
    ts = [0.1, 0.3, 1.0, 10.0, 100.0]
        
    for t in ts:
        psf = gaussfft(deltafcn(128, 128), t, plot_kernel=plot_kernel)
        
        showgrey(psf, False)
        plt.title(r"$PSF, t={}$".format(t))
        plt.show()
        cov = variance(psf)

        print(f"PSF, t={t}")
        print(cov)
        var = np.mean(np.diag(cov))
        print(f"Mean variance: {var}")
        print("_"*50)
    
def ex8():
    ts = [1, 4, 16, 64, 256]
    
    fig = plt.figure()
    
    for i, t in enumerate(ts):
        ax = fig.add_subplot(1, len(ts), i+1)
        img = np.load(f"images-npy/tower256.npy")
        psf = gaussfft(img, t, False)
        
        showgrey(psf, False)
        ax.set_title(r"$t={}$".format(t))
        cov = variance(psf)

        print(f"PSF, t={t}")
        print(cov)
        var = np.mean(np.diag(cov))
        print(f"Mean variance: {var}")
        print("_"*50)
    plt.show()
    
def ex9(filter_type="gaussian"):
    office = np.load("Images-npy/office256.npy")
    add = gaussnoise(office, 16)
    sap = sapnoise(office, 0.1, 255)
    
    # Gaussian smoothing
    
    if filter_type == "gaussian":
        ts = [0.1, 0.3, 1.0, 10.0, 100.0]
        fig = plt.figure()

        for i, t in enumerate(ts):
            gaussian_noise_img = gaussfft(add, t, False)
            snp_denoised = gaussfft(sap, t, False)
        
            ax = fig.add_subplot(len(ts), 2, 2*i+1)
            showgrey(gaussian_noise_img, False)
            ax.set_title(r"Gaussian noise, $t={}$".format(t))
            
            ax = fig.add_subplot(len(ts), 2, 2*i+2)
            showgrey(snp_denoised, False)
            ax.set_title(r"S&P noise, $t={}$".format(t))
            
        fig.suptitle("Gaussian smoothing")
        plt.show()
        
    elif filter_type == "median":
        window_sizes = [1, 2, 4, 16, 32]#, 64, 128, 256]
        
        fig = plt.figure()

        for i, window_size in enumerate(window_sizes):
            gaussian_noise_img = medfilt(add, window_size)
            snp_denoised = medfilt(sap, window_size)
        
            ax = fig.add_subplot(len(window_sizes), 2, 2*i+1)
            showgrey(gaussian_noise_img, False)
            ax.set_title(r"Gaussian noise, $w_s={}$".format(window_size))
            
            ax = fig.add_subplot(len(window_sizes), 2, 2*i+2)
            showgrey(snp_denoised, False)
            ax.set_title(r"S&P noise, $w_s={}$".format(window_size))
            
        fig.suptitle("Median filtering")
        plt.show()
        
    elif filter_type == "ideal_low_pass":
        cutoffs = [0.01, 0.05, 0.1, 0.25, 0.5]
        
        
        gaussian_noise_img = ideal(add, 0.1, plot_filter=True)

        fig = plt.figure()

        for i, cutoff in enumerate(cutoffs):
            gaussian_noise_img = ideal(add, cutoff)
            snp_denoised = ideal(sap, cutoff)
        
            ax = fig.add_subplot(len(cutoffs), 2, 2*i+1)
            showgrey(gaussian_noise_img, False)
            ax.set_title(r"Gaussian noise, $f_c={}$".format(cutoff))
            
            ax = fig.add_subplot(len(cutoffs), 2, 2*i+2)
            showgrey(snp_denoised, False)
            ax.set_title(r"S&P noise, $f_c={}$".format(cutoff))
            
        fig.suptitle("Ideal Low Pass filtering")
        plt.show()
        
def ex10(ideal_filter=False):
    img = np.load("Images-npy/phonecalc256.npy")
    smoothimg = img
    N = 5
    f = plt.figure()
    f.subplots_adjust(wspace=0, hspace=0)
    for i in range(N):
        if i>0: # generate subsampled versions
            img = rawsubsample(img)
            if ideal_filter:
                smoothimg = ideal(smoothimg, 1/8)
            else: 
                smoothimg = gaussfft(smoothimg, 10)
            smoothimg = rawsubsample(smoothimg)
        f.add_subplot(2, N, i + 1)
        showgrey(img, False)
        f.add_subplot(2, N, i + N + 1)
        showgrey(smoothimg, False)
    plt.show()
        

if __name__ == '__main__':

    exercise = 10

    # Q1-6
    if exercise == 1:
        ex1()
    # Q7-9
    elif exercise == 2:
        # ex2()
        ex2(use_log=True)
    # Q10
    elif exercise == 3:
        ex3()
    # Q11
    elif exercise == 4:
        ex4()
    # Q12
    elif exercise == 5:
        ex5()
    # Q13
    elif exercise == 6:
        ex6()
    # Q14-15
    elif exercise == 7:
        ex7()
    # Q16
    elif exercise == 8:
        ex8()
    # Q17-18
    elif exercise == 9:
        ex9(filter_type="ideal_low_pass")
    # Q19-20
    elif exercise == 10:
        ex10(ideal_filter=True)