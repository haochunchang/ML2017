import numpy as np
import pandas as pd
import sys, random, pickle
import feature
import matplotlib.pyplot as plt

def train_lr(train, iternum=1, lamb=1e-4):
    '''
    Training W using train_data in train
    
    Loss Function: Mean square error
    model: # of order of the regression line
    Return trained W matrix
    '''
    
    # Model initialization 
    W = np.zeros(190,)
    y = []
    X = []
    std = train.labelstd
    mu = train.labelmu
    
    for i in range(len(train)):
        y.append(train.get_label(i))
        X.append(train[i]) 
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    
    lr = 1e-1
    pre_grad = 0
    iteration = iternum
    for epoch in range(1, iteration+1):    
        grad = 0
        loss = 0
        total_loss = 0
        # Run through all training data
        y_hat = np.dot(X, W)
        delta = y_hat - y    
        loss = np.dot(delta, delta.T) + lamb * np.dot(W, W.T)     
        grad = 2 * (np.dot(X.T, delta) + lamb * W)
 
        pre_grad += grad ** 2
        rms_g = np.sqrt(pre_grad + 1e-8)
            
        # Update parameters
        W -= (lr / rms_g) * grad
        rmse = np.sqrt(loss / len(train))
        print(rmse.sum())
    
    print("Training loss: %f" % rmse.sum()) 
    '''
    XTX = np.linalg.pinv(np.matmul(X.T, X))
    W = np.matmul(np.matmul(XTX, X.T), y)
    y_hat = np.dot(X, W)    

    loss = ((y_hat - y) ** 2) / len(train)    
    rmse = np.sqrt(loss)
    
    print("Training loss: %f" % rmse.sum()) 
    '''
    return W
 
def test_lr(W, test, outfilepath, train):
    '''
    Use trained W to predict test_X.csv and output result file.
    '''
    test = feature.TestFeature(test, train.mu, train.std)
    test = test.scaling()
    test.add_bias()
    with open(outfilepath, 'w') as o:
        o.write("id,value\n")
        for i in range(len(test)):
            total_y = 0
            for j in range(len(W)):
                total_y += np.dot(test[i], W[j])
            y = total_y / len(W)   
            y = str(float(y * train.labelstd + train.labelmu))
            o.write("id_"+str(i)+","+y)
            o.write("\n")
    print("Testing result stored in %s" % outfilepath)

def validate(W, val_f, val_l, train):
    loss = 0
    for i in range(len(val_f)):
        y = np.multiply(W, val_f[i]).sum()
        y_hat = val_l[i]
        loss += ((y - y_hat) ** 2) / len(train)
    total_loss = loss * train.labelstd + train.labelmu
    print("Validation loss: %f" % total_loss)
    return total_loss
 
def lr_main(train, test, outfilepath):
    '''
    Linear Regression main function
    
    train: filepath of train.csv
    test: filepath of test_X.csv
    outfilepath: filepath of predicted_result.csv
    '''
    # Load in data & preprocessing
    training_data = pd.read_csv(train, sep=",", encoding="big5") 

    # Training
    W_best = []
    for i in range(1):
        train = feature.Feature(training_data)
        train = train.scaling()
        train.add_bias()
        #train, _ = train.bagging(train)
        W = train_lr(train, iternum=50000)
        W_best.append(W)
    
    with open("./model/W_best.pkl", "wb") as o:
        pickle.dump(W_best, o)
    
    # Testing and output result
    test = pd.read_csv(test, sep=",", header=None)
    test_lr(W_best, test, outfilepath, train)
    
if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[2]
    outfilepath = sys.argv[3]
    lr_main(train, test, outfilepath)
