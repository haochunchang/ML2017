import pandas as pd
import numpy as np
import sys, pickle
import feature

def activate(x, threshold):
    return (x >= threshold).astype(np.int)

def sigmoid(x):
    sigmoid = 1.0 / (1.0+np.exp(-x))
    return sigmoid

def logi_test(W, xtest, outfilepath):
    
    # Loading in testing data
    test_data = pd.read_csv(xtest)
    test = feature.Xfeature(test_data, None).preprocess()
    
    # Normalization
    test = test.bucketize(20)
    test.add_bias()

    with open(outfilepath, 'w') as o:
        o.write("id,label\n")
        z1 = sigmoid(np.dot(test, W[0]))
        y = activate(z1, 0.5)
        for i in range(len(test)):   
            o.write(str(i+1)+","+str(y[i]))
            o.write("\n")
    print("Testing result stored in %s" % outfilepath)
     
def logi_main(xtest, outfilepath):
    
    # Load in trained model
    with open("./model/W_sbl.pkl", "rb") as w:
        W_trained = pickle.load(w)
  
    # Testing and output prediction file
    logi_test(W_trained, xtest, outfilepath)

if __name__ == "__main__":
    xtest = sys.argv[5]
    outfilepath = sys.argv[6]
    logi_main(xtest, outfilepath)
