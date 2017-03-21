import numpy as np
import pandas as pd
import sys, random, pickle, math
import feature
import matplotlib.pyplot as plt

def train_lr(train, iternum=1, lamb=1e-3):
    '''
    Training W using train_data in train
    
    Loss Function: Mean square error
    model: # of order of the regression line
    Return trained W matrix
    '''
    
    # Model initialization 
    W = np.zeros((163,))
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
    for epoch in range(iteration):    
        y_hat = np.dot(X, W)
        delta = y_hat - y    
        loss = (np.dot(delta, delta.T) + lamb * np.absolute(W).sum())  
        grad = 2 * (np.dot(X.T, delta)) + lamb * np.sign(W)
 
        pre_grad += grad ** 2
        rms_g = np.sqrt(pre_grad + 1e-8)
            
        # Update parameters
        W -= (lr / rms_g) * grad
        rmse = np.sqrt(loss.sum() / len(train))
        print(rmse)
    
    print("Training loss: %f" % rmse) 
    '''
    XTX = np.linalg.pinv(np.matmul(X.T, X))
    W = np.matmul(np.matmul(XTX, X.T), y)
    y_hat = np.dot(X, W)    

    loss = ((y_hat - y) ** 2) / len(train)    
    rmse = np.sqrt(loss.sum())
    
    print("Training loss: %f" % rmse) 
    '''
    return W
 
def test_lr(W, test, d, outfilepath, train):
    '''
    Use trained W to predict test_X.csv and output result file.
    '''
    test = feature.TestFeature(test, train.mu, train.std)
    test = test.scaling(d)
    test.add_bias()

    x = []
    for i in range(len(test)):
        x.append(test[i])
    x = np.array(x, dtype=np.float64)

    with open(outfilepath, 'w') as o:
        o.write("id,value\n")
        total_y = 0
        for j in range(len(W)):
            total_y += np.dot(x, W[j])
        Y = total_y / len(W)
        
        for i in range(len(Y)):
            y = str(float(Y[i]))
            o.write("id_"+str(i)+","+y)
            o.write("\n")
    print("Testing result stored in %s" % outfilepath)

def validate(W, val_f, val_l, lamb):
    loss = 0
    for i in range(len(val_f)):
        y_hat = np.dot(val_f[i], W)
        y = val_l[i]
        loss += ((y_hat - y) ** 2 + lamb * np.absolute(W).sum())
    total_loss = np.sqrt(loss.sum() / len(val_f))
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
    for i in range(30):
        train = feature.Feature(training_data)
        train, d = train.scaling()
        train.add_bias()
        train, _ = train.bagging(train)
        W = train_lr(train, iternum=100000, lamb=1e-2)
        W_best.append(W)

    with open("./model/W_best.pkl", "wb") as o:
        pickle.dump(W_best, o)
    with open("./model/W_best.csv", "w") as o:
        for W in W_best:
            for line in W:
                o.write(str(line))
                o.write("\n")
    
    # Testing and output result
    test = pd.read_csv(test, sep=",", header=None)
    test_lr(W_best, test, d, outfilepath, train)
    
if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[2]
    outfilepath = sys.argv[3]
    lr_main(train, test, outfilepath)
