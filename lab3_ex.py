import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from Functions import *

from kmeans_example import *

def ex1():
    pass

if __name__ == '__main__':
    
    exercise = 1
    
    imgs = {
        "orange": Image.open('Images-jpg/orange.jpg'),
        "tiger1": Image.open('Images-jpg/tiger1.jpg'),
        "tiger2": Image.open('Images-jpg/tiger2.jpg'),
        "tiger3": Image.open('Images-jpg/tiger3.jpg'),
        }
    
    if exercise==1:
        sel = "orange"
        
        if sel=="orange":
            kmeans_example(imgs[sel], K=10, L=50,
                           scale_factor=0.5, image_sigma=1.0,
                           boundary=True)
        

    elif exercise==2:
        pass