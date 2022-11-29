import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from Functions import *

from kmeans_example import *
from mean_shift_example import *

IMGS = {
        "orange": Image.open('Images-jpg/orange.jpg'),
        "tiger1": Image.open('Images-jpg/tiger1.jpg'),
        "tiger2": Image.open('Images-jpg/tiger2.jpg'),
        "tiger3": Image.open('Images-jpg/tiger3.jpg'),
        }


def ex1_(convergence, sel, scale_factor, image_sigma, iters=10):
    for k in [10, 20, 30]:
        val = 0
        for i in range(iters):
            val += kmeans_example(IMGS[sel], K=k, L=None,
                    scale_factor=scale_factor, image_sigma=image_sigma,
                    verbose=i==0, seed=i)
            
            if not convergence:
                break
            
        if convergence:
            print(f"For {k} clusters, we converge in {val/iters} iters on average")

def ex1(sel, convergence=False):
    if sel=="orange":
        ex1_(convergence, sel, scale_factor=0.5, image_sigma=1.0)
        ex1_(convergence, sel, scale_factor=0.2, image_sigma=1.0)
        ex1_(convergence, sel, scale_factor=0.5, image_sigma=5.0)
    
    
def ex2():
    
    sel="orange"
    seed = np.random.randint(100)
    for k in range(2, 16):
        kmeans_example(IMGS[sel], K=k, L=None,
                        scale_factor=1, image_sigma=1.0,
                        verbose=True, seed=seed)
            

def ex3():
    
    sel="tiger1"
    # With orange params
    kmeans_example(IMGS[sel], K=5, L=23,
                        scale_factor=1, image_sigma=1.0,
                        boundary=True, verbose=True)
    
    kmeans_example(IMGS[sel], K=5, L=None,
                        scale_factor=1, image_sigma=1.0,
                        boundary=True, verbose=True)
    
    kmeans_example(IMGS[sel], K=14, L=86,
                        scale_factor=1, image_sigma=1.0,
                        boundary=False, verbose=True)
    
    kmeans_example(IMGS[sel], K=14, L=None,
                        scale_factor=1, image_sigma=1.0,
                        boundary=False, verbose=True)
    
   
    for k in range(15, 20):
        kmeans_example(IMGS[sel], K=k, L=None,
                            scale_factor=1, image_sigma=1.0,
                            boundary=False, verbose=True)


def ex4(sel, test=False):
    
    if test:
        # kmeans
        kmeans_example(IMGS[sel], K=16, L=None,
                       boundary=False, seed=4321)
        
        # Testing bandwiths
    
        mean_shift_example(IMGS[sel],
                        spatial_bandwidth=10,
                        colour_bandwidth=20)
        
        # color bandwith
      
        mean_shift_example(IMGS[sel],
                        spatial_bandwidth=10,
                        colour_bandwidth=3)
    
        mean_shift_example(IMGS[sel],
                        spatial_bandwidth=10,
                        colour_bandwidth=40)
        
        # spatital bandwith
       
        mean_shift_example(IMGS[sel],
                        spatial_bandwidth=3,
                        colour_bandwidth=20)
    

        mean_shift_example(IMGS[sel],
                        spatial_bandwidth=20,
                        colour_bandwidth=20)
    
    elif sel=="orange":
        mean_shift_example(IMGS[sel],
                        spatial_bandwidth=35,
                        colour_bandwidth=8,
                        num_iterations=100)
    elif sel=="tiger1":
        mean_shift_example(IMGS[sel],
                        spatial_bandwidth=5,
                        colour_bandwidth=70,
                        num_iterations=100)
        






if __name__ == '__main__':
    
    exercise = 4
        
    if exercise==1:    
        ex1("orange", convergence=True)

    elif exercise==2:
        ex2()

    elif exercise==3:
        ex3()
    
    elif exercise==4:
        # ex4("orange", test=True)
        # ex4("orange")
        ex4("tiger1")