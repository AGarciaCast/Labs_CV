import sys
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from lab3 import kmeans_segm
from Functions import mean_segments, overlay_bounds

def kmeans_example(img, K=10, L=50, scale_factor=0.5, image_sigma=1.0,
                   boundary=True, seed=14, verbose=True):

    img = img.resize((int(img.size[0]*scale_factor), int(img.size[1]*scale_factor)))
    
    h = ImageFilter.GaussianBlur(image_sigma)
    I = np.asarray(img.filter(ImageFilter.GaussianBlur(image_sigma))).astype(np.float32)
    segm, centers, iters = kmeans_segm(I, K, L, seed)
    
    if verbose:
        Inew = mean_segments(img, segm)
        if boundary:
            Inew = overlay_bounds(img, segm)

        img = Image.fromarray(Inew.astype(np.ubyte))
        plt.imshow(img)
        plt.title(f"K-mean, w/ K={K}, iters={iters}, sf={scale_factor}, sigma={image_sigma}")
        plt.axis('off')
        plt.show()
    
    return iters

if __name__ == '__main__':
    img = Image.open('Images-jpg/orange.jpg')
    print(kmeans_example(img, K=20, L=None, boundary=True))
