#You are only allowed to use np.linalg.svd or np.linalg.eig to calculate the eigenfaces. 
#Libraries such as scikit-learn are prohibited.

import os, sys, glob
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def pca_main(data_path):
    
    img_path = './part1/'
    if not os.path.isdir(img_path):
        os.mkdir(img_path)
 
    # Load in face expression data into a (100, 64*64) numpy matrix
    filelist = sorted(glob.glob(data_path+'/[A-J]0[0-9].bmp'))
    image_list = [misc.imread(filename) for filename in filelist]   

    image_list = np.array(image_list).reshape(len(image_list), 64*64)
    
    # Center dataset
    mu = image_list.mean(axis=0)
    imgs = image_list - mu
    
    # Compute SVD
    U, s, V = np.linalg.svd(imgs.T, full_matrices=False)

#-----------------Report 1.1---------------------------------------------
    
    misc.imsave(os.path.join(img_path, 'mu_face.png'), mu.reshape(64,64))
    plot_faces((3, 3), U.T[:9].reshape(9, 64, 64))

#-----------------Report 1.2----------------------------------------------
    
    # Plot 100 original faces
    plot_faces((10, 10), image_list.reshape(len(image_list), 64, 64))
    
    # Project faces onto Top 5 components and reconstruct them
    evecs = U.T[:5] #(5, 4096)
    transformed = np.dot(imgs, evecs.T) #(100, 5)
    recover = np.dot(evecs.T, transformed.T).T + mu
    
    # Plot reconstructed faces
    plot_faces((10, 10), recover.reshape(100, 64, 64)) 

#-----------------Report 1.3----------------------------------------------
    for k in range(U.T.shape[0]):
        evecs = U.T[:k]
        transformed = np.dot(imgs, evecs.T)
        recover = np.dot(evecs.T, transformed.T).T + mu
        error = ((image_list - recover) ** 2 / image_list.shape[0]).sum()   
        print(np.sqrt(error))
        if np.sqrt(error) < 0.01:
            print(k)
            break 

def plot_faces(shape=(3, 3), images=None):
    
    plt.figure(0)
    axes = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            axes.append(plt.subplot2grid(shape, (i,j)))
    
    for i in range(shape[0]*shape[1]):
        axes[i].imshow(images[i], cmap=plt.cm.gray)
        axes[i].axis('off')
    plt.show()

if __name__ == "__main__":
    pca_main(sys.argv[1])
