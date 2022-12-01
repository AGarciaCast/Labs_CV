import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from Functions import *
from gaussfft import gaussfft
from scipy.stats import multivariate_normal

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
    
    if len(image.shape)==3:
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
    # kmeans_segm(image, K, 10)
    image = image / 255
    I = np.reshape(image, (-1,3)).astype(np.float32)
    flat_mask = np.reshape(mask, (mask.shape[0]*mask.shape[1]))  
    # Store all pixels for which mask=1 in a Nx3 matrix
    
    c = I[flat_mask>0, :]
    N = c.shape[0]
    # Randomly initialize the K components using masked pixels
    
    seg, mu, _ = kmeans_segm(c, K, L=L)
    w = np.array([np.mean(seg==k) for k in range(K)])
    sigma = np.ones((K, 3, 3))* 0.01
    
    P = np.zeros((N, K))
    # print(c.shape,
    #       mu.shape,
    #       w.shape,
    #       sigma.shape,
    #       P.shape)
    
    # Iterate L times
    for _ in range(L):
        
        # Expectation: Compute probabilities P_ik using masked pixels
        for k in range(K):
            p_x_c = multivariate_normal(
                mean=mu[k], cov=sigma[k], allow_singular=True)
            
            P[:, k] = w[k]*p_x_c.pdf(c)
        
        for i in range(N):
            P[i, :] = P[i, :]/(sum(P[i, :]) + 1e-5)
            
        # Maximization: Update weights, means and covariances using masked pixels
        P_class = np.sum(P, axis=0)
        
        w = P_class/N
        mu = (P.T @ c)/P_class[:, None]
        
        for k in range(K):
            P_k = P[:, k]
            P_k = P_k[:, None]
            aux = c - mu[k]
            sigma[k] = (aux.T @ (aux * P_k))/P_class[k]


    # Compute probabilities p(c_i) in Eq.(3) for all pixels I.  
    res = np.zeros(I.shape[0])
    
    for k in range(K):
            p_x_c = multivariate_normal(
                mean=mu[k], cov=sigma[k], allow_singular=True)
            
            res += w[k]*p_x_c.pdf(I)
        
    
    return res


