import numpy as np
import sys, pickle
import utils

from keras import backend as K
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

def train(x_train, y_train, model_path):
          
    (_, x_train, _) = utils.read_data(x_train, True)
    '''
    (_, x_test, _) = utils.read_data('data/test_data.csv', False)
    
    tfidf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer())])

    tfidf.fit(x_train + x_test)
    '''
    with open('models/tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    # Split validation data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=6)

    # Fit model
    x_train = tfidf.transform(x_train).todense()
    clf = OneVsRestClassifier(LinearSVC(C=0.03, class_weight='balanced'))
    clf.fit(x_train, y_train)
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
   
    x_val = tfidf.transform(x_val).todense() 
    f1(clf, x_val, y_val)

def f1(model, x_val, y_val):
        
    thresh = 0.5
    
    Y_pred = model.predict(x_val)
    Y_pred_thresh = (Y_pred > thresh).astype('int')       

    for i in range(Y_pred_thresh.shape[0]):
        if Y_pred_thresh[i].all() == 0:
            max_idx = Y_pred[i].argmax()
            Y_pred_thresh[i][max_idx] = 1
    score = f1_score(y_val, Y_pred_thresh, average='samples')
    print('f1_score: {}'.format(score))
 

def main(train_filepath):
    
    #x_train = np.loadtxt('x_train_tfidf.txt')
    model_id = 0
    x_train, y_train = utils.preprocess(train_filepath, './data/test_data.csv', save_token=False)
    train(train_filepath, y_train, 'models/svc{}.pkl'.format(model_id))    

if __name__ == "__main__":
    main(sys.argv[1])

