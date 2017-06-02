import numpy as np
import pandas as pd
import pickle, sys, os
import test_best, test_matplus, test_dnn

def ensemble(data_dir, outfilepath):

    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    best = test_best.main(data_dir)
    matplus = test_matplus.main(data_dir)
    dnn = test_dnn.main(data_dir)

    weights = [0.3, 0.2, 0.5]
    pred = weights[0] * best['Rating'] + weights[1] * matplus['Rating'] + weights[2] * dnn['Rating']
    
    sub = pd.DataFrame()
    sub['TestDataID'] = test['TestDataID']
    sub['Rating'] = pred
    sub.to_csv(outfilepath, index=False) 
    print('Test result stored in {}'.format(outfilepath))
    
if __name__ == "__main__":
    ensemble(sys.argv[1], sys.argv[2]) 
