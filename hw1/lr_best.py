import numpy as np
import pandas as pd
import sys, random, pickle
import feature
import matplotlib.pyplot as plt

def train_lr(train, model = 1):
    '''
    Training W using train_data in train
    
    Loss Function: Mean square error
    model: # of order of the regression line
    Return trained W matrix
    '''
    # Model initialization 
    W = np.zeros((1,163))

    lr = 10e-3
    iteration = 10000
    for epoch in range(1, iteration+1):    
        grad = 0 
        loss = 0
        for i in range(len(train)):
            # Compute predicted value
            y = np.dot(W, train[i])  
            y_hat = train.get_label(i)

            loss += ((y - y_hat) ** 2) / (2 * len(train))
            grad += ((y - y_hat) * train[i]) / len(train)  
                
            # Update parameters
        W = W - lr * grad
    print("Training loss: %f" % np.sqrt(loss))
    
    return W

def validate(W, val_feature, val_label):
    '''
    Validate W using val data.
    Return validation average error
    '''
    loss = 0
    for i in range(len(val_feature)):
        y = np.dot(W, val_feature[i]) 
        y_hat = val_label[i]
        loss += ((y_hat - y) ** 2) / (2 * len(train))
    
    print("Validation loss: %f" % np.sqrt(loss)) 
    return loss  

def test_lr(W, test, outfilepath):
    '''
    Use trained W to predict test_X.csv and output result file.
    '''
    test = feature.TestFeature(test)
    test.flatten()
    with open(outfilepath, 'w') as o:
        o.write("id,value\n")
        for i in range(len(test)):
            y = np.dot(W, test[i])    
            y = str(y)[2:-2]
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
    
    # Feature & label extraction
    train = pd.read_csv(train, sep=",", encoding="big5")
    train = feature.Feature(train)

    # Flatten feature and add bias into (163,)
    train.flatten()
    
    val_f, val_l = train.sample_val(240)
    W_best = train_lr(train, model=1)
    validate(W_best, val_f, val_l) 
    
    
    with open("./model/W_best.pkl", "wb") as o:
        pickle.dump(W_best, o)
    
    # Testing and output result
    test = pd.read_csv(test, sep=",", header=None)
    test_lr(W_best, test, outfilepath)

if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[2]
    outfilepath = sys.argv[3]
    lr_main(train, test, outfilepath)
