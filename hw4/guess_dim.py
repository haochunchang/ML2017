import sys, os, pickle
import math
import numpy as np

def predict_dim(data, index):

    ref = np.loadtxt('train_mmv.txt')
    ref = ref[:, 1] 
    mvar = (data.std(axis=1) ** 2).mean() 
    
    dim = np.absolute(ref - mvar).argmin() + 1
    print(index, dim)
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
