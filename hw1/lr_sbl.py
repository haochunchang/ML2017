import numpy as np
import pandas as pd
import sys, random, pickle
import feature
import matplotlib.pyplot as plt


def test_lr(W, test, outfilepath, train):
    '''
    Use trained W to predict test_X.csv and output result file.
    '''
    test = feature.TestFeature(test, train.mu, train.std)
    with open(outfilepath, 'w') as o:
        o.write("id,value\n")
        for i in range(len(test)):
            y = np.multiply(W, test[i]).sum() * train.labelstd + train.labelmu    
            y = str(y)
            o.write("id_"+str(i)+","+y)
            o.write("\n")
    print("Testing result stored in %s" % outfilepath)
        
def lr_main(train, test, outfilepath):
    '''
    Linear Regression main function
    
    train: filepath of train.csv
    test: filepath of test_X.csv
    outfilepath: filepath of predicted_result.csv
    '''
    
    training_data = pd.read_csv(train, sep=",", encoding="big5") 
    
    train = feature.Feature(training_data)   
    with open("./model/W.pkl", "rb") as w:
        W_best = pickle.load(w)
 
    # Testing and output result
    test = pd.read_csv(test, sep=",", header=None)
    test_lr(W_best, test, outfilepath, train)

if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[2]
    outfilepath = sys.argv[3]
    lr_main(train, test, outfilepath)
