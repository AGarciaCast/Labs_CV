import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from Functions import *

from kmeans_example import *
from mean_shift_example import *
from norm_cuts_example import *
from graphcut_example import *

IMGS = {
        "orange": Image.open('Images-jpg/orange.jpg'),
        "tiger1": Image.open('Images-jpg/tiger1.jpg'),
        "tiger2": Image.open('Images-jpg/tiger2.jpg'),
        "tiger3": Image.open('Images-jpg/tiger3.jpg'),
        }


def ex1(sel, convergence=False):
    def ex1_(scale_factor, image_sigma, iters=10):
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
    # kmeans_example(IMGS[sel], K=5, L=23,
    #                     scale_factor=1, image_sigma=1.0,
    #                     boundary=True, verbose=True)
    
    # kmeans_example(IMGS[sel], K=5, L=None,
    #                     scale_factor=1, image_sigma=1.0,
    #                     boundary=True, verbose=True)
    
    # kmeans_example(IMGS[sel], K=14, L=86,
    #                     scale_factor=1, image_sigma=1.0,
    #                     boundary=False, verbose=True)
    
    kmeans_example(IMGS[sel], K=16, L=None,
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
        

def ex5(sel, test=False):
    if test:
        norm_cuts_example(IMGS[sel], 
                        colour_bandwidth=20.0,
                        radius=1,
                        ncuts_thresh=0.10, 
                        min_area=200, 
                        max_depth=12)
        
        # bandwith
        norm_cuts_example(IMGS[sel], 
                        colour_bandwidth=8.0,
                        radius=1,
                        ncuts_thresh=0.10, 
                        min_area=200, 
                        max_depth=12)
        
        norm_cuts_example(IMGS[sel], 
                        colour_bandwidth=30.0,
                        radius=1,
                        ncuts_thresh=0.10, 
                        min_area=200, 
                        max_depth=12)
        
        # threshold
        norm_cuts_example(IMGS[sel], 
                        colour_bandwidth=20.0,
                        radius=1,
                        ncuts_thresh=0.01, 
                        min_area=200, 
                        max_depth=12)
        
        norm_cuts_example(IMGS[sel], 
                        colour_bandwidth=20.0,
                        radius=1,
                        ncuts_thresh=0.50, 
                        min_area=200, 
                        max_depth=12)
        
        # area
        norm_cuts_example(IMGS[sel], 
                        colour_bandwidth=20.0,
                        radius=1,
                        ncuts_thresh=0.10, 
                        min_area=300, 
                        max_depth=12)
        
        norm_cuts_example(IMGS[sel], 
                        colour_bandwidth=20.0,
                        radius=1,
                        ncuts_thresh=0.10, 
                        min_area=50, 
                        max_depth=12)
        
        # depth
        norm_cuts_example(IMGS[sel], 
                        colour_bandwidth=20.0,
                        radius=1,
                        ncuts_thresh=0.10, 
                        min_area=200, 
                        max_depth=30)
        
        norm_cuts_example(IMGS[sel], 
                        colour_bandwidth=20.0,
                        radius=1,
                        ncuts_thresh=0.10, 
                        min_area=200, 
                        max_depth=4)
        
    elif sel == "orange":
        norm_cuts_example(IMGS[sel], 
                        colour_bandwidth=35.0,
                        radius=3,
                        ncuts_thresh=0.01, 
                        min_area=10, 
                        max_depth=4)
        
        norm_cuts_example(IMGS[sel], 
                        colour_bandwidth=20.0,
                        radius=3,
                        ncuts_thresh=0.03, 
                        min_area=10, 
                        max_depth=4)
    
    elif sel == "tiger1":
        norm_cuts_example(IMGS[sel], 
                        colour_bandwidth=20,
                        radius=2,
                        ncuts_thresh=0.03, 
                        min_area=10, 
                        max_depth=6)
        
        norm_cuts_example(IMGS[sel], 
                        colour_bandwidth=20,
                        radius=2,
                        ncuts_thresh=0.03, 
                        min_area=30, 
                        max_depth=6)
        
        norm_cuts_example(IMGS[sel], 
                        colour_bandwidth=20,
                        radius=30,
                        ncuts_thresh=0.1, 
                        min_area=10, 
                        max_depth=6)


def ex6(sel, test=False):
    if test:
        # efect area
        graphcut_example(IMGS[sel],
                        area=[80, 110, 570, 300], 
                        K = 16, alpha=8.0, sigma=20.0)
        
        graphcut_example(IMGS[sel],
                        area=[120, 84, 234, 260], 
                        K = 16, alpha=8.0, sigma=20.0)
        
        # efect alpha
        graphcut_example(IMGS[sel],
                        area=[80, 110, 570, 300], 
                        K = 16, alpha=20.0, sigma=20.0)
        
        graphcut_example(IMGS[sel],
                        area=[80, 110, 570, 300], 
                        K = 16, alpha=1.0, sigma=20.0)
        
        # efect sigma
        graphcut_example(IMGS[sel],
                        area=[80, 110, 570, 300], 
                        K = 16, alpha=8.0, sigma=50.0)
        
        graphcut_example(IMGS[sel],
                        area=[80, 110, 570, 300], 
                        K = 16, alpha=8.0, sigma=5.0)
        
        
    elif sel == "tiger1":
        graphcut_example(IMGS[sel],
                        area=[80, 110, 570, 300], 
                        K = 16, alpha=35.0, sigma=4.0)
        
        graphcut_example(IMGS[sel],
                        area=[120, 84, 234, 260], 
                        K = 16, alpha=35.0, sigma=4.0)
    
    elif sel == "tiger3":
        graphcut_example(IMGS[sel],
                        area=[252, 134, 512, 258], 
                        K = 16, alpha=40.0, sigma=8.0)
        
        
def ex7(sel):
    for k in range(1, 16):
        if sel == "tiger1":
            graphcut_example(IMGS[sel],
                            area=[80, 110, 570, 300], 
                            K = k, alpha=35.0, sigma=4.0)
        elif sel == "tiger3":
            graphcut_example(IMGS[sel],
                        area=[252, 134, 512, 258], 
                        K = k, alpha=40.0, sigma=8.0)  


if __name__ == '__main__':
    
    exercise = 6
        
    if exercise==1:    
        # Q1
        ex1("orange", convergence=True)

    elif exercise==2:
        # Q3
        ex2()

    elif exercise==3:
        # Q4
        ex3()
    
    elif exercise==4:
        # Q5, Q6
        # ex4("orange", test=True)
        # ex4("orange")
        ex4("tiger1")
        
    elif exercise==5:
        # Q7, Q8, Q9, Q10
        # ex5("orange", test=True)
        # ex5("orange")
        ex5("tiger1")
    
    elif exercise==6:
        # Q11, Q13
        # ex6("tiger1", test=True)
        ex6("tiger1")
        ex6("tiger3")
    
    elif exercise==7:
        # Q12
        # ex7("tiger1")
        ex7("tiger3")