import numpy as np
import sys, pickle
import utils

   
def predict(model_id, test_filepath, outfilepath):
    # Loading in trained model 
    model_path = 'models/svc{}.pkl'.format(model_id)
    
    with open('models/tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    
    with open('best_tag_mapping.pkl', 'rb') as f:
        tag_list = pickle.load(f)

    (_, x_test, _) = utils.read_data(test_filepath, False)
    x_test = tfidf.transform(x_test).todense()

    # Predict
    thresh = 0.5
    
    Y_pred = clf.predict(x_test)
    with open(outfilepath,'w') as output:
        print ('\"id\",\"tags\"',file=output)
        Y_pred_thresh = (Y_pred > thresh).astype('int')       

        for index,labels in enumerate(Y_pred_thresh):
            labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
            if ' '.join(labels) == '':
                i = Y_pred[index].argmax()
                labels = [tag_list[i]]
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)
    
    print("Testing result stored in %s" % outfilepath)

if __name__ == "__main__":
    predict(sys.argv[3], sys.argv[1], sys.argv[2])
