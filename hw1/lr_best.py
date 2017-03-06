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
    W = np.zeros((19,9))
    mu = train.labelmu
    std = train.labelstd

    lr = 10e-2
    iteration = 10000
    pre_grad = 0
    total_loss = 0
    for epoch in range(1, iteration+1):    
        grad = 0
        loss = 0
        total_loss = 0
        train = train.shuffle()
        for i in range(len(train)):
            # Compute predicted value
            y = np.multiply(W, train[i]).sum()  
            y_hat = train.get_label(i)

            loss = ((y - y_hat) ** 2) / len(train) 
            grad = (2 * (y - y_hat) * train[i]) / len(train) 
        
            # Update parameters
            pre_grad += grad ** 2
            W -= (lr / np.sqrt(pre_grad)) * grad
            total_loss += loss
        rmse = np.sqrt(total_loss * std + mu)
        print(rmse)
    print("Training loss: %f" % rmse)
    
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

def test_lr(W, test, outfilepath, train):
    '''
    Use trained W to predict test_X.csv and output result file.
    '''
    test = feature.TestFeature(test, mu=train.mu, std=train.std)
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
    
    # Feature & label extraction
    train = pd.read_csv(train, sep=",", encoding="big5")
    train = feature.Feature(train)
    
    #val_f, val_l = train.sample_val(240)
    W_best = train_lr(train, model=1)
    #validate(W_best, val_f, val_l) 
    
    with open("./model/W_best_alltrain_temp.pkl", "wb") as o:
        pickle.dump(W_best, o)
    
    # Testing and output result
    test = pd.read_csv(test, sep=",", header=None)
    test_lr(W_best, test, outfilepath, train)

if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[2]
    outfilepath = sys.argv[3]
    lr_main(train, test, outfilepath)
