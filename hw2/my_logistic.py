import pandas as pd
import numpy as np
import sys, pickle
import feature

def logi_main(raw_train, raw_test, outfilepath):
    
    # Loading in Training data
    train_data = pd.read_csv(raw_train, header=None)

    # Extract Labels (32561,)
    label_extract = lambda x: 1 if x else 0
    labels = np.array([label_extract(x == ' >50K') for x in train_data.iloc[:, 14]])

    # Hashing base categorical features
    ''' 
    train = feature.Xfeature(train_data, labels).preprocess()
    
    # Normalization
    train = train.normalize()
    train.add_bias()  
    
    # Model initialization
    W = np.zeros((107,))    

    # Training
    W_trained = logi_train(W, train_data, labels, epoch=1000, batch_size=20, lr=1e-3)
    
    # Save model
    with open("./model/W_my_logistic.pkl", "wb") as o:
        pickle.dump(W_trained, o)
    with open("./model/W_my_logistic.csv", "w") as o:
        for W in W_trained:
            o.write(str(W))
            o.write("\n")
    
    # Loading in testing data
    test_data = pd.read_csv(xtest)
    test = feature.Xfeature(test_data, None).preprocess()
    
    # Normalization
    test = test.normalize()
    test.add_bias()

    # Testing and output prediction file
    logi_test(W_trained, xtest, outfilepath)
    '''

def activate(x):
    sigmoid = 1.0 / (1.0+np.exp(-x))
    return (sigmoid >= 0.5).astype(np.int)

def dsigmoid(x):
    ds = np.exp(-x) / ((1.0+np.exp(-x)) ** 2)
    return ds
 
def logi_train(W, train_data, labels, epoch=1000, batch_size=20, lr=1e-1, lamb=1e-3):
    
    nbatch = len(train) // batch_size + 1
    pre_grad = 0
    rho = 0.9
    for i in range(1, epoch+1):
        err = 0
        for j in range(nbatch):
            
            batch_x, batch_y = train.sample(batch_size)
            y_hat = activate(np.dot(batch_x, W))
            grad = -np.dot(batch_x.T, (batch_y - y_hat)) # L = cross entropy
            loss = np.absolute(batch_y - y_hat).sum()            

            pre_grad = rho * pre_grad + (1-rho) * (grad ** 2)
            W -= (lr / np.sqrt(pre_grad+1e-8)) * grad
            err += loss
        if i % 100 == 0:
            print('Training accuracy after %d epoch: %f' %(i, 1 - err / len(train)))

    return W 

def logi_test(W, test, outfilepath):
    
    with open(outfilepath, 'w') as o:
        o.write("id,label\n")
        for i in range(len(test)):
            y = activate(np.dot(test[i], W))   
            o.write(str(i+1)+","+str(y))
            o.write("\n")
    print("Testing result stored in %s" % outfilepath)
     
if __name__ == "__main__":
    raw_train = sys.argv[1]
    raw_test = sys.argv[2]
    outfilepath = sys.argv[3]
    logi_main(raw_train, raw_test, outfilepath)
