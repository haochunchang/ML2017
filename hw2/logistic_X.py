import pandas as pd
import numpy as np
import sys, pickle
import feature

def activate(x, threshold):
    return (x >= threshold).astype(np.int)

def sigmoid(x):
    sigmoid = 1.0 / (1.0+np.exp(-x))
    return sigmoid

def dsigmoid(x):
    ds = np.exp(-x) / ((1.0+np.exp(-x)) ** 2)
    return ds
 
def logi_train(W, train_data, labels, epoch=1000, batch_size=20, lr=1e-1, lamb=1e-3):
    
    train = feature.Xfeature(train_data, labels).preprocess()
    
    # Normalization
    train = train.bucketize()
    train.add_bias()  

    nbatch = len(train) // batch_size + 1
    pre_grad1, pre_grad2 = (0, 0)
    for i in range(1, epoch+1):
        err = 0
        for j in range(nbatch):           
            
            batch_x, batch_y = train.sample(batch_size)
            z = sigmoid(np.dot(batch_x, W[0])) 
            y_hat = activate(z, 0.5)

            # L = cross entropy
            grad1 = (-1) * np.dot(batch_x.T, (batch_y - z)) / batch_size + 2 * lamb * W[0] / batch_size
            
            # Record previous gradients
            pre_grad1 += grad1 ** 2
            
            # Update parameters
            W[0] -= (lr / np.sqrt(pre_grad1+1e-8)) * grad1

            # total number of misclassification
            loss = np.absolute(batch_y - y_hat).sum()
            err += loss
        if i % 100 == 0:
            print('Training accuracy after %d epoch: %f' %(i, 1 - err / len(train)))

    return W 

def logi_test(W, xtest, outfilepath):
    
    # Loading in testing data
    test_data = pd.read_csv(xtest)
    test = feature.Xfeature(test_data, None).preprocess()
    
    # Normalization
    test = test.bucketize()
    test.add_bias()

    with open(outfilepath, 'w') as o:
        o.write("id,label\n")
        z1 = sigmoid(np.dot(test, W[0]))
        y = activate(z1, 0.5)
        for i in range(len(test)):   
            o.write(str(i+1)+","+str(y[i]))
            o.write("\n")
    print("Testing result stored in %s" % outfilepath)
     
def logi_main(xtrain, ytrain, xtest, outfilepath):
    
    # Loading in Training data
    train_data = pd.read_csv(xtrain)
    labels = pd.read_csv(ytrain, header=None)
    
    # Model initialization
    W = []
    W.append(np.zeros((107,)))    
 
    # Training
    W_trained = logi_train(W, train_data, labels, batch_size=30, epoch=2000, lr=1e-1, lamb=1e-4)
    
    # Save model
    with open("./model/W_logistic_X.pkl", "wb") as o:
        pickle.dump(W_trained, o)
    with open("./model/W_logistic_X.csv", "w") as o:
        for W in W_trained:
            o.write(str(W))
            o.write("\n")
       
    # Testing and output prediction file
    logi_test(W_trained, xtest, outfilepath)

if __name__ == "__main__":
    xtrain = sys.argv[1]
    ytrain = sys.argv[2]
    xtest = sys.argv[3]
    outfilepath = sys.argv[4]
    logi_main(xtrain, ytrain, xtest, outfilepath)
