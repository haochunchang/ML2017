import numpy as np
import pandas as pd
import pickle, sys, os
import test_best, test_matplus, test_dnn

def ensemble(data_dir, outfilepath):
    mf = ['prediction_mf.csv']
    rnn = ['prediction_matplus.csv']
    dnn = ['prediction_dnn.csv']

    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    test_best.main(data_dir, mf[0])
    test_matplus.main(data_dir, rnn[0])
    test_dnn.main(data_dir, dnn[0])

    best = pd.read_csv(mf[0])
    matplus = pd.read_csv(rnn[0])
    dnn = pd.read_csv(dnn[0])

    weights = [0.4, 0.2, 0.4]
    pred = weights[0] * best['Rating'] + weights[1] * matplus['Rating'] + weights[2] * dnn['Rating']
    
    sub = pd.DataFrame()
    sub['TestDataID'] = test['TestDataID']
    sub['Rating'] = pred
    sub.to_csv(outfilepath, index=False) 
    print('Test result stored in {}'.format(outfilepath))
    
if __name__ == "__main__":
    ensemble(sys.argv[1], sys.argv[2]) 
