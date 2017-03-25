import pandas as pd
import numpy as np
import sys, pickle, math
import feature

def activate(x):
    sigmoid = 1.0 / (1.0+math.exp(-x))
    if sigmoid >= 0.5:
        return 0
    else:
        return 1
def v_activate(x):
    sigmoid = 1.0 / (1.0+np.exp(-x))
    return (sigmoid < 0.5).astype(np.int)


def generative_W(train_data, labels):
    
    train = feature.Xfeature(train_data, labels).preprocess()
    
    # Normalization
    train = train.normalize()

    # Maximize Likelihood
    x_class1 = np.array([train[i] for i in range(len(train)) if train.get_label(i) == 0])
    x_class2 = np.array([train[i] for i in range(len(train)) if train.get_label(i) == 1])
    n1 = x_class1.shape[0]
    n2 = x_class2.shape[0]
     
    mu1 = x_class1.mean(axis=0)
    mu2 = x_class2.mean(axis=0)
    
    delta1 = x_class1 - mu1
    delta2 = x_class2 - mu2    
    
    cov1 = (np.dot(delta1.T, delta1)) / n1
    cov2 = (np.dot(delta2.T, delta2)) / n2
    cov = (n1 / len(train)) * cov1 + (n2 / len(train)) * cov2
    cov_inv = np.linalg.inv(cov)
    
    # Construct W from maximum likelihood 
    w = np.dot((mu1-mu2), cov_inv)
    b = (np.dot(np.dot(mu2.T, cov_inv), mu2) - np.dot(np.dot(mu1.T, cov_inv), mu1)) / 2 + np.log(n1/n2)
   
    train_x, train_y = train.sample(len(train))
    err = np.absolute(v_activate(np.dot(train_x, w) + b) - train_y).sum()
    print('Training accuracy: %f' %(1 - err / len(train)))
    return w, b 
    
def generate_test(W, b, xtest, outfilepath):
    
    # Loading in testing data
    test_data = pd.read_csv(xtest)
    test = feature.Xfeature(test_data, None).preprocess()
    
    # Normalization
    test = test.normalize()

    with open(outfilepath, 'w') as o:
        o.write("id,label\n")
        for i in range(len(test)): 
            y = activate(np.dot(W, test[i]) + b)
            o.write(str(i+1)+","+str(y))
            o.write("\n")
    print("Testing result stored in %s" % outfilepath)
     
def generate_main(xtrain, ytrain, xtest, outfilepath):
    
    # Loading in Training data
    train_data = pd.read_csv(xtrain)
    labels = pd.read_csv(ytrain, header=None)
    
    # Training
    W, b = generative_W(train_data, labels)
    
    # Save model
    with open("./model/W_generate_X.pkl", "wb") as o:
        pickle.dump((W, b), o)
       
    # Testing and output prediction file
    generate_test(W, b, xtest, outfilepath)

if __name__ == "__main__":
    xtrain = sys.argv[1]
    ytrain = sys.argv[2]
    xtest = sys.argv[3]
    outfilepath = sys.argv[4]
    generate_main(xtrain, ytrain, xtest, outfilepath)
