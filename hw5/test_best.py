import numpy as np
import sys, pickle
import keras
from keras.models import model_from_json
from keras.preprocessing import sequence, text
from sklearn.metrics import f1_score
import utils

def nn_test(test_filepath, outfilepath):
    
    # Load in pre-trained tokenizer.
    with open('best_text_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
   
    with open('best_tag_mapping.pkl', 'rb') as f:
        tag_list = pickle.load(f)

    (_, x_test, _) = utils.read_data(test_filepath, False)
    seqs = tokenizer.texts_to_sequences(x_test)    
    x_test = tokenizer.sequences_to_matrix(seqs, mode='tfidf')
    #x_test = sequence.pad_sequences(seqs, maxlen=306)
    
    # Loading in trained model 
    with open("models/best_model.json", "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights('models/best.h5')

    # Predict
    thresh = 0.4
    Y_pred = model.predict(x_test, batch_size=128, verbose=1)
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
    nn_test(sys.argv[1], sys.argv[2]) 
