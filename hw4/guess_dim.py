import sys, pickle
import math
import numpy as np
from sklearn.decomposition import KernelPCA, FactorAnalysis
from sklearn import metrics
from sklearn import manifold
from sklearn import neighbors
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

def predict_dim(data, index):

    portion = len(data) // 10
    data = data[:portion]
    logMLE = 0
    dim = 1 
    
    k = 20
    nbrs = neighbors.NearestNeighbors(n_neighbors=k, algorithm='kd_tree', n_jobs=-1).fit(data)     
    with open('./data/precomputed/index{}_{}neighbor_graph.pkl'.format(index, k), 'wb') as g:
        pickle.dump(nbrs, g)
    
    Tk = nbrs.kneighbors(data)[0][:,k-1]
    for j in range(1, k-1):
        logMLE += np.log(Tk / nbrs.kneighbors(data)[0][:,j])
 
    logMLE = np.mean(logMLE / (k - 2))
    dim = math.floor(1 / logMLE)
    print('Best dimesion of index {} is {}.'.format(index, dim))

    return math.log(dim)

def main(data_path, outfilepath):
    data = np.load(data_path)

    with open(outfilepath, 'w') as o:
        o.write("SetId,LogDim\n")  
        for i in range(200):
            Logdim = predict_dim(data[str(i)], i)   
            o.write(str(i)+","+str(Logdim))
            o.write("\n")
    print("Testing result stored in %s" % outfilepath)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
