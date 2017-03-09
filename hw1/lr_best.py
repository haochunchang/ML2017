import numpy as np
import pandas as pd
import sys, random, pickle
import feature
import matplotlib.pyplot as plt

def train_lr(train, train2, model = 1):
    '''
    Training W using train_data in train
    
    Loss Function: Mean square error
    model: # of order of the regression line
    Return trained W matrix
    '''
    # Model initialization 
    W = np.zeros((19,9))
    W2 = np.zeros((19,9))
    mu = train.labelmu
    std = train.labelstd

    lr = 10e-6
    iteration = 1000
    pre_grad = 0
    pre_grad2 = 0
    rho = 0.9
    total_loss = 0
    for epoch in range(1, iteration+1):    
        grad = 0
        grad2 = 0
        loss = 0
        total_loss = 0
        for i in range(len(train)):
            # Compute predicted value
            y = np.multiply(W, train[i]).sum() + np.multiply(W2, train2[i]).sum()
            y_hat = train.get_label(i)
        
            loss = ((y - y_hat) ** 2) / len(train) 
            
            grad = 2 * ((y - y_hat) * train[i]) / len(train) 
            grad2 = 2 * ((y - y_hat) * train2[i]) / len(train)     
            
        # Update parameters
            pre_grad = rho * pre_grad + (1-rho) * (grad ** 2)
            pre_grad2 = rho * pre_grad2 + (1-rho) * (grad2 ** 2)
            W -= (lr / np.sqrt(pre_grad)) * grad
            W2 -= (lr / np.sqrt(pre_grad2)) * grad2
            total_loss += loss
        rmse = np.sqrt(total_loss * std + mu)
        print(rmse)
    print("Training loss: %f" % rmse)
    
    return W, W2

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

def test_lr(W, W2, test, test2, train, outfilepath):
    '''
    Use trained W to predict test_X.csv and output result file.
    '''
    std = train.labelstd
    mu = train.labelmu
    with open(outfilepath, 'w') as o:
        o.write("id,value\n")
        for i in range(len(test)):
            y = np.multiply(W, test[i]).sum()  
            y = y + np.multiply(W2, test2[i]).sum()
            y = str(y * std + mu)
            o.write("id_"+str(i)+","+y)
            o.write("\n")
    print("Testing result stored in %s" % outfilepath)
        
def test_ensemble(W_lst, test, test2, train, outfilepath):
    '''
    Input: trained models
    Average models' predicted value as answer
    Output: result_best.csv
    '''
    std = train.labelstd
    mu = train.labelmu
    
    with open(outfilepath, 'w') as o:
        o.write("id,value\n")
        for i in range(len(test)):
            y_avg = 0
            for j in range(len(W_lst)):
                y = np.multiply(W_lst[j][0], test[i]).sum() + np.multiply(W_lst[j][1], test2[i]).sum()
                y_avg += y
            y_avg = y_avg / len(W_lst)
            y = str(y * std + mu)
            o.write("id_"+str(i)+","+y)
            o.write("\n")
    print("Testing result stored in %s" % outfilepath)
        
def lr_main(train, test, outfilepath):
    '''
    Regression main function
    
    train: filepath of train.csv
    test: filepath of test_X.csv
    outfilepath: filepath of predicted_result.csv
    '''
    
    # Load in training data 
    train_data = pd.read_csv(train, sep=",", encoding="big5")
    #train_data2 = train_data.applymap(lambda x: np.square(float(x)) if x.isnumeric() else x)

    # Random sample training data to train 3 quadratic models.
    W_lst = []
    for i in range(3):
        train = feature.Feature(train_data)
        train2 = feature.Feature(train_data).square()

        vf, vl = train.sample_val(len(train) // 20)
        vf2, vl2 = train2.sample_val(len(train2) // 20)
        W, W2 = train_lr(train, train2)
        W_lst.append((W, W2))
    
    with open("./model/W_best.pkl", "wb") as o:
        pickle.dump(W_lst, o)

    # Load in testing data
    test_data = pd.read_csv(test, sep=",", header=None)
    test = feature.TestFeature(test_data, mu=train.mu, std=train.std)
    test2 = feature.TestFeature(test_data, mu=train.mu, std=train.std).square()

    # Testing and output result
    test_ensemble(W_lst, test, test2, train, outfilepath)

if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[2]
    outfilepath = sys.argv[3]
    lr_main(train, test, outfilepath)
