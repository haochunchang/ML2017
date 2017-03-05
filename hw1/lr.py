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
    W = []
    for i in range(model):
        W.append(np.zeros((1,163)))

    lr = 10e-3
    pre_grad = [0 for i in range(model)]
    rms_g = [0 for i in range(model)]
    rho = 0.9 # decaying coefficient
    iteration = 10000
    for epoch in range(1, iteration+1):    
        grad = [0 for i in range(model)]
        loss = 0
        total_loss = 0
        # Run through all training data
        for i in range(len(train)):
            y = 0
            y_hat = train.get_label(i)
            # Compute predicted value
            for j in range(1, len(W) + 1):
                y = y + np.dot(W[j-1], train[i] ** j)  
                grad[j-1] += ((y - y_hat) * (train[i] ** j)) / len(train) 
            loss += ((y - y_hat) ** 2) / (2 * len(train))
            
        # RMSprop
        for j in range(1, len(W) + 1):
            pre_grad[j-1] = rho * pre_grad[j-1] + (1 - rho) * (grad[j-1] ** 2)
            rms_g[j-1] = np.sqrt(pre_grad[j-1] + 10e-8)
                
            # Update parameters
            W[j-1] = W[j-1] - (lr / rms_g[j-1]) * grad[j-1]
        
        total_loss = np.sqrt(loss)
        print(total_loss)
        #his_e.append(epoch)
        #his_l.append(total_loss)
    
    #plt.plot(his_e, his_l)
    #plt.show()        
    print("Training loss of model %d: %f" % (model, total_loss)) 
    return W

def validate(W, val_feature, val_label):
    '''
    Validate W using val data.
    Return validation average error
    '''
    loss = 0
    for i in range(len(val_feature)):
        y = 0
        y_hat = val_label[i]
        for j in range(1, len(W) + 1):
            y = y + np.dot(W[j - 1], val_feature[i] ** j) 
        loss += ((y - y_hat) ** 2) / (2 * len(train))

    print("Validation loss: %f" % np.sqrt(loss)) 
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
            y = 0
            for j in range(1, len(W) + 1):
                y = y + np.dot(W[j - 1], test[i] ** j)    
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
    
    training_data = pd.read_csv(train, sep=",", encoding="big5") 
    
    train = feature.Feature(training_data)    
    train.flatten()
    val_f, val_l = train.sample_val(240)
    W_best = train_lr(train, model=1)
    validate(W_best, val_f, val_l)
    
    ''' 
    # Training
    best_err = 10e8
    for i in range(1, 2):
        
        # Feature & label extraction
        train = feature.Feature(training_data)
        
        # Flatten feature and add bias into (163,)
        train.flatten()

        val_feature, val_label = train.sample_val(240)
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
