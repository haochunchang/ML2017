import sys, os, pickle
import glob
import numpy as np
from scipy import misc
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVR as SVR
from generate_data import get_eigenvalues

def predict_dim(data):
    
    # Load model
    with open('model/svr.pkl', 'rb') as f:
        model = pickle.load(f)

    # predict
    dims = []
    test = []
    for i in range(data.shape[0]):
        evals = get_eigenvalues(data[i])
        test.append(evals)
    test = np.array(test)
    dims = model.predict(test)

    return np.exp(dims)

def main():
    
    # Load in hand rotation data
    filelist = sorted(glob.glob('./data/hand/*.png'))
    image_list = [misc.imread(filename) for filename in filelist]   
    imgs = np.array(image_list).reshape(len(image_list), 480*512)

    dims = predict_dim(imgs)
    print(dims.mean())

if __name__ == "__main__":
    main()
