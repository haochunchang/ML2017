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
    # For plotting
    history_e = []
    history_l = []
    
    # Model initialization 
    W = np.zeros((1,163))

    lr = 10
    iteration = 10000
    pre_grad = 0
    #while delta_loss > 10e-6:
    for epoch in range(1, iteration+1):    
        grad = 0 
        loss = 0
        for i in range(len(train)):
            # Compute predicted value
            y = np.dot(W, train[i])  
            y_hat = train.get_label(i)

            loss += (y - y_hat) ** 2
            grad += (y - y_hat) * train[i] * 2  
                
            # Update parameters
        pre_grad += grad ** 2
        W = W - (lr / np.sqrt(pre_grad)) * grad
        #print(loss / len(train))
        # Plot loss v.s. epoch
        history_e.append(epoch)
        history_l.append(float(loss / len(train)))
    print("Training loss: %f" % (loss / len(train)))
    plt.plot(history_e, history_l)
    plt.show()
    
    return W

def validate(W, val_feature, val_label):
    '''
    Validate W using val data.
    Return validation average error
    '''
    loss = 0
    for i in range(len(val_feature)):
        y = 0
        for j in range(1, len(W) + 1):
                y = y + np.dot(W[j - 1], val_feature[i] ** j) 
        y_hat = val_label[i]
        loss = loss + (y_hat - y) ** 2
    loss = loss / len(val_feature)
    print(loss) 
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
            if y < 0:
                y = str(np.array([0]))[1:-1]
            else:
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
    
    W_best = train_lr(train, model=1)
    '''
    # Training    
    best_err = 10e8
    for i in range(1, 2):
        val_feature, val_label = train.sample_val(len(train) // 5)
        W = train_lr(train, model=i)

        # Validation    
        err = validate(W, val_feature, val_label)
        if err < best_err:
            best_err = err
            W_best = W
    
    with open("./model/W.pkl", "wb") as o:
        pickle.dump(W_best, o)
    '''
    # Testing and output result
    test = pd.read_csv(test, sep=",", header=None)
    test_lr(W_best, test, outfilepath)

if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[2]
    outfilepath = sys.argv[3]
    lr_main(train, test, outfilepath)
