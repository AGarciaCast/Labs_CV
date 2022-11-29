import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from Functions import *
from gaussfft import gaussfft


def kmeans_segm(image, K, L, seed = 42):
    """
    Implement a function that uses K-means to find cluster 'centers'
    and a 'segmentation' with an index per pixel indicating with 
    cluster it is associated to.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        seed - random seed
    Output:
        segmentation: an integer image with cluster indices
        centers: an array with K cluster mean colors
    """ 
    img = image.reshape((-1, 3))
    
    np.random.seed(seed)
    unique = np.unique(img, axis=0)
    idx = np.random.choice(unique.shape[0], K, replace=False)
    centers = unique[idx]
    
    old_centers= np.zeros_like(centers)
    num_iters = 0
    
    while L is None or num_iters<L:
        
        old_centers = centers.copy()
        dist = distance_matrix(img, centers)
        segmentation = np.argmin(dist, axis=1)
        
        for k in range(K):
            
            cluster = img[segmentation==k]
            if len(cluster)>0:
                centers[k] = np.mean(cluster, axis=0)
        
        num_iters += 1
        
        if np.max(np.max(abs(centers - old_centers))) < 1e-3:
            break
    
    segmentation = segmentation.reshape((image.shape[0], image.shape[1]))

    return segmentation, centers, num_iters


def mixture_prob(image, K, L, mask):
    """
    Implement a function that creates a Gaussian mixture models using the pixels 
    in an image for which mask=1 and then returns an image with probabilities for
    every pixel in the original image.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        mask - an integer image where mask=1 indicates pixels used 
    Output:
        prob: an image with probabilities per pixel
    """ 
    return prob



