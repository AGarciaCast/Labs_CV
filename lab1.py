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
    # fftwave(9, 5)
    # fftwave(17, 9)
    # fftwave(17, 121)
    # fftwave(5, 1)
    # fftwave(125, 1)

def ex2(use_log=True):
    F = np.concatenate([np.zeros((56,128)), np.ones((16,128)), np.zeros((56,128))])
    G = F.T
    H = F + 2*G
    fig = plt.figure()
    _ = fig.add_subplot(3, 3, 1)
    showgrey(F, False)
    _ = fig.add_subplot(3, 3, 2)
    showgrey(G, False)
    _ = fig.add_subplot(3, 3, 3)
    showgrey(H, False)
    Fhat = fft2(F)
    Ghat = fft2(G)
    Hhat = fft2(H)

    _ = fig.add_subplot(3, 3, 4)
    showgrey(np.log(1 + np.abs(Fhat)), False)

    _ = fig.add_subplot(3, 3, 5)
    showgrey(np.log(1 + np.abs(Ghat)), False)

    _ = fig.add_subplot(3, 3, 6)
    showgrey(np.log(1 + np.abs(Hhat)), False)

    _ = fig.add_subplot(3, 3, 7)
    shift_Fhat = fftshift(Fhat)
    if use_log:
        showgrey(np.log(1 + np.abs(shift_Fhat)), False)
    else:
        showgrey(np.abs(shift_Fhat), False)

    _ = fig.add_subplot(3, 3, 8)
    shift_Ghat = fftshift(Ghat)
    if use_log:
        showgrey(np.log(1 + np.abs(shift_Ghat)), False)
    else:
        showgrey(np.abs(shift_Ghat), False)

    _ = fig.add_subplot(3, 3, 9)
    shift_Hhat = fftshift(Hhat)
    if use_log:
        showgrey(np.log(1 + np.abs(shift_Hhat)), False)
        # showfs(Hhat, False)
    else:
        showgrey(np.abs(shift_Hhat), False)

    plt.show()

def ex3():
    F = np.concatenate([np.zeros((56, 128)), np.ones((16, 128)), np.zeros((56, 128))])
    G = F.T
    Fhat = fft2(F)
    Ghat = fft2(G)
    fig = plt.figure()
    _ = fig.add_subplot(2, 3, 1)
    showgrey(F, False)
    _ = fig.add_subplot(2, 3, 2)
    showgrey(G, False)
    _ = fig.add_subplot(2, 3, 3)
    showgrey(F * G, False)
    _ = fig.add_subplot(2, 3, 4)
    showfs(fft2(F), False)
    _ = fig.add_subplot(2, 3, 5)
    showfs(fft2(G), False)
    _ = fig.add_subplot(2, 3, 6)
    showfs(fft2(F*G), False)
    plt.show()
    # Stackoverflow
    # showfs(fftshift(convolve2d(fftshift(Fhat)/128, fftshift(Ghat)/128, mode="same")))
    
    Xf = np.fft.fft2(F)
    Yf = np.fft.fft2(G)
    showfs(Yf)
    N = Xf.size    # or Yf.size since they must have the same size
    Zf = np.vstack([Yf, Yf, Yf])
    Zf = np.hstack([Zf, Zf, Zf])
    showfs(Zf)
    print(Zf.shape)
    # why we need inverse shift
    conv = np.fft.ifftshift(convolve2d(Xf, Zf, mode="same")/128**2)
    showfs(conv)



def ex4():
    F_ = np.concatenate([np.zeros((56, 128)), np.ones((16, 128)), np.zeros((56, 128))])
    showgrey(F_)
    showfs(fft2(F_))
    
    F = np.concatenate([np.zeros((60, 128)), np.ones((8, 128)), np.zeros((60, 128))]) * \
        np.concatenate([np.zeros((128, 48)), np.ones((128, 32)), np.zeros((128, 48))], axis=1)
    
    G = F.T
    Fhat = fft2(F)
    Ghat = fft2(G)
    fig = plt.figure()
    _ = fig.add_subplot(2, 3, 1)
    showgrey(F, False)
    _ = fig.add_subplot(2, 3, 2)
    showgrey(G, False)
    _ = fig.add_subplot(2, 3, 3)
    showgrey(F * G, False)
    _ = fig.add_subplot(2, 3, 4)
    showfs(fft2(F), False)
    _ = fig.add_subplot(2, 3, 5)
    showfs(fft2(G), False)
    _ = fig.add_subplot(2, 3, 6)
    showfs(fft2(F*G), False)
    
    plt.show()
    
    
    # alpha = [30, 60, 90]
    # for angle in alpha:
    #     G = rot(F, angle)
    #     showgrey(G)
    #     Ghat = fft2(G)
    #     showfs(Ghat)
    #     Hhat = rot(fftshift(Ghat), -angle)
    #     showgrey(np.log(1 + abs(Hhat)))


    # images = ['phonecalc128.npy', 'office128.npy']
    # for loc in images:
    #     img = np.load(f"images-npy/{loc}")
    #     showgrey(img)
    #     showgrey(pow2image(img))
    #     showgrey(randphaseimage(img))

if __name__ == '__main__':

    exercise = 3

    if exercise == 1:
        ex1()
    elif exercise == 2:
        # ex2()
        ex2(use_log=True)
    elif exercise == 3:
        ex3()
    elif exercise == 4:
        ex4()