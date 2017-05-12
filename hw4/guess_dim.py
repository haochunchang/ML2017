import sys, os, pickle
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors as kNN
from sklearn.svm import LinearSVR as SVR
from generate_data import get_eigenvalues

def predict_dim(data):
    
    # Train model
    npzfile = np.load('train_data.npz')
    X = npzfile['X']
    y = npzfile['y']
    model = SVR(C=1.5)
    model.fit(X, y)   

    with open('model/svr.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # predict
    dims = []
    test = []
    for i in range(200):
        evals = get_eigenvalues(data[str(i)])
        test.append(evals)
    test = np.array(test)
    dims = model.predict(test)

    return dims

def main(data_path, outfilepath):
    data = np.load(data_path)
    dims = predict_dim(data)
    
    with open(outfilepath, 'w') as o:
        o.write("SetId,LogDim\n")  
        for i, Logdim in enumerate(dims):
            o.write(str(i)+","+str(Logdim))
            o.write("\n")
    print("Testing result stored in %s" % outfilepath)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
