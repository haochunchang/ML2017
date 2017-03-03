import numpy as np
import pandas as pd
import sys, random
import feature

def train_lr(train, model = 1):
    '''
    Training W using train_data in train
    
    Loss Function: Mean square error
    model: # of order of the regression line
    Return trained W matrix
    '''
    # Model initialization 
    W = []
    for i in range(model):
        W.append(np.zeros((1,163)))

    pre_grad = 0
    learning_rate = 1
    iter_num = 10000
    for iteration in range(iter_num):    
        grad = 0
        loss = 0
        for i in range(len(train)):
            y = 0
            for j in range(1, len(W) + 1):
                y = y + np.dot(W[j - 1], train[i] ** j)  
            y_hat = train.get_label(i)
            loss = loss + (y_hat - y) ** 2
            grad = grad - 2 * (y_hat - y) * train[i]
        
        # Update parameters
        pre_grad += grad ** 2 
        W = W - (learning_rate / np.sqrt(pre_grad)) * grad
    print(loss / len(train))
    return W

def validate(W, val_feature, val_label):
    '''
    Validate W using val data.
    Return validation average error
    '''
    loss = 0
    for j in range(len(val_feature)):
        y = np.dot(W, val_feature[j])
        y_hat = val_label[j]
        loss = loss + (y_hat - y) ** 2
    loss = loss / len(val_feature)
    
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
            o.write("id_"+str(i)+","+str(y)[1:-1]+"\n")
        

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

    # Training & LOOV
    best_err = 10e8
    for i in range(1, 4):
        val_feature, val_label = train.sample_val(len(train) // 3)
        W = train_lr(train, model=i)

        # Validation    
        err = validate(W, val_feature, val_label)
        if err < val_err:
            best_err = err
            W_best = W
    
    # Testing and output result
    test = pd.read_csv(test, sep=",", header=None)
    test_lr(W_best, test, outfilepath)

if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[2]
    outfilepath = sys.argv[3]
    lr_main(train, test, outfilepath)
