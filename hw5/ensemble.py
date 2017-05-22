import numpy as np
import utils
import pickle, sys
import svc_test
import test_best, test_rnn_ens

def ensemble(test_filepath, outfilepath):
    svc_preds = ['prediction_svc{}.csv'.format(i) for i in range(3)]
    dnn = ['prediction_dnn.csv']
    rnn = ['prediction_rnnsbl.csv']
    
    test_best.nn_test(test_filepath, dnn[0])
    test_rnn_ens.nn_test(test_filepath, rnn[0])
    for i in range(3): 
        svc_test.predict(i, test_filepath, svc_preds[i])

    labels = []
    for filename in svc_preds + rnn + dnn:
        labels.append(utils.get_prediction_labels(filename))

    weights = [0.1, 0.1, 0.1, 0.3, 0.4]
    labels = np.array(labels)
    pred_sum = np.zeros((labels.shape[1], labels.shape[2])) #(1234, 38)
    for i in range(len(weights)):
        pred_sum += weights[i] * labels[i]
    prediction = pred_sum 

    utils.save_prediction_results(outfilepath, prediction)
    print('Test result stored in {}'.format(outfilepath))
    
if __name__ == "__main__":
    ensemble(sys.argv[1], sys.argv[2]) 
