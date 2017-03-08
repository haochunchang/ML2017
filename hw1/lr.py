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
    #his_e = []
    #his_l = []
    
    # Model initialization 
    W = np.zeros((19,9))
    std = train.labelstd
    mu = train.labelmu
    
    lr = 10e-5
    pre_grad = 0
    rho = 0.9 # decaying coefficient
    iteration = 1000
    for epoch in range(1, iteration+1):    
        grad = 0
        loss = 0
        total_loss = 0
        train = train.shuffle()
        # Run through all training data
        for i in range(len(train)):
            y_hat = train.get_label(i)
            y = np.multiply(W, train[i]).sum()  
            grad = (2 * ((y - y_hat)) * train[i]) / len(train) 
            loss = ((y - y_hat) ** 2) / len(train)
             
            pre_grad = rho * pre_grad + (1 - rho) * (grad ** 2)
            rms_g = np.sqrt(pre_grad + 10e-8)
            
            # Update parameters
            W -= (lr / rms_g) * grad
            total_loss += loss
        total_loss = np.sqrt(total_loss * std + mu)
        print(total_loss)
        #his_e.append(epoch)
        #his_l.append(total_loss)
    
    #plt.plot(his_e, his_l)
    #plt.show()        
    print("Training loss of model %d: %f" % (model, total_loss)) 
    return W
 
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
    W_best = train_lr(train, model=1)
    '''
    with open("./model/W.pkl", "wb") as o:
        pickle.dump(W_best, o)
    '''
    # Testing and output result
    test = pd.read_csv(test, sep=",", header=None)
    test_lr(W_best, test, outfilepath, train)

if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[2]
    outfilepath = sys.argv[3]
    lr_main(train, test, outfilepath)
