import pandas as pd
import numpy as np
import sys, pickle
import feature, math
import matplotlib.pyplot as plt

def activate(x, threshold):
    return (x >= threshold).astype(np.int)

def sigmoid(x):
    sigmoid = 1.0 / (1.0+np.exp(-x))
    return np.clip(sigmoid, 0.000000000001, 0.999999999999)

def dsigmoid(x):
    ds = np.exp(-x) / ((1.0+np.exp(-x)) ** 2)
    return ds
 
def logi_train(W, train_data, labels, epoch=1000, batch_size=20, lr=1e-1, lamb=1e-3):
    
    train = feature.Xfeature(train_data, labels).preprocess()
    
    # Bucketization of continuous attributes to different number of bins.
    train = train.bucketize(0, 20)
    train = train.bucketize(1, 20)
    train = train.bucketize(3, 20)
    train = train.bucketize(4, 20)
    train = train.bucketize(5, 20)
    train.add_bias()  

    #---Sample 1/10 training data as Validation data---
    #val_f, val_l = train.sample_val(len(train) // 10)
    #--------------------------------------------------

    nbatch = len(train) // batch_size
    pre_grad1, pre_grad2 = (0, 0)
    for i in range(1, epoch+1):
        err = 0
        train.sample_times = 0
        train = train.shuffle()
        for j in range(nbatch):           
            batch_x, batch_y = train.sample(batch_size)
            #batch_x, batch_y = train.get()
 
            z = sigmoid(np.dot(batch_x, W[0])) 
            y_hat = activate(z, 0.5)

            # L = cross entropy
            grad1 = (-1) * np.dot(batch_x.T, (batch_y - z)) + lamb * np.sign(W[0])
            
            # Record previous gradients
            pre_grad1 += grad1 ** 2
            
            # Update parameters
            W[0] -= (lr / np.sqrt(pre_grad1+1e-8)) * grad1

            # total number of misclassification
            loss = np.absolute(batch_y - y_hat).sum()
            err += loss
             
        if i % 10 == 0:
            print('Training accuracy after %d epoch: %f' %(i, 1 - err / len(train)))
              
 
    #---Validation---------------------------------
    #y_hat = activate(sigmoid(np.dot(val_f, W[0])),0.5)
    #val_loss = np.absolute(val_l - y_hat).sum() / val_f.shape[0]
    #print('Validation accuracy: %f' %(1 - val_loss)) 
    #----------------------------------------------    
    
    return W#, val_loss 

def logi_test(W, xtest, outfilepath):
    
    # Loading in testing data
    test_data = pd.read_csv(xtest)
    test = feature.Xfeature(test_data, None).preprocess()
    
    # Normalization
    test = test.bucketize(0, 20)
    test = test.bucketize(1, 20)
    test = test.bucketize(3, 20)
    test = test.bucketize(4, 20)
    test = test.bucketize(5, 20)
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
    
    # Training
    #lamb, loss = [], []
    #best_val = 1e8
    #for l in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]: 
        
    # Model initialization
    W = []
    W.append(np.zeros((107,)))    

    W_trained = logi_train(W, train_data, labels, batch_size=10, epoch=1000, lr=1e-1, lamb=1e-4)
       # lamb.append(math.log10(l))
       # loss.append(val_loss)    
       # if val_loss < best_val:
       #     W_best = W_trained
       #     best_val = val_loss

    # Plot
    #plt.scatter(lamb, loss)
    #plt.show()
    #plt.close()

    W_best = W_trained
    # Save model
    with open("./model/W_logistic_X.pkl", "wb") as o:
        pickle.dump(W_best, o)
    with open("./model/W_logistic_X.csv", "w") as o:
        for W in W_best:
            o.write(str(W))
            o.write("\n")
       
    # Testing and output prediction file
    logi_test(W_best, xtest, outfilepath)

if __name__ == "__main__":
    xtrain = sys.argv[1]
    ytrain = sys.argv[2]
    xtest = sys.argv[3]
    outfilepath = sys.argv[4]
    logi_main(xtrain, ytrain, xtest, outfilepath)
