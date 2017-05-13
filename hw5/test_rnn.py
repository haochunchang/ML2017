import numpy as np
import sys, pickle
import keras
from keras.models import model_from_json
from keras.preprocessing import sequence, text

def predict(rnn, x_test, batch_size=128):

    out = rnn.predict_proba(x_test, batch_size=batch_size, verbose=1) 
    best_thresholds = np.loadtxt('best_threshold.txt')
    y_pred = np.array([[1 if out[i,j]>=best_thresholds[j] else 0 for j in range(out.shape[1])] for i in range(len(x_test))])    

    # Load in tokenizer
    with open('tag_tokenizer.pkl', 'rb') as f:
        y_token = pickle.load(f)

    # Translate class number to tags
    trans = y_token.word_index
    predicted_tags = []
    
    for i in range(y_pred.shape[0]):
        for idxs in np.nonzero(y_pred[i]):
            if len(idxs) == 0:
                predicted_tags.append([[key for key, val in trans.items() if val == out[i].argmax()+1]])
            else:
                predicted_tags.append([[key for key, val in trans.items() if val == idx+1] for idx in idxs])

    return predicted_tags

def nn_test(test_filepath, outfilepath):
    
    # Loading in testing data & Preprocessing
    idx = []
    para = []
    with open(test_filepath, 'r', encoding='latin-1') as f:
        for line in f:
            if line.split(',')[0] == 'id':
                continue
            else:
                line = line.split(',')
                ix = int(line[0])
                para_text = ''.join(line[1:])
                idx.append(ix)
                para.append(para_text)

    x_tokenizer = text.Tokenizer()
    x_tokenizer.fit_on_texts(para)
    seqs = x_tokenizer.texts_to_sequences(para)    
    x_test = sequence.pad_sequences(seqs)

    # Loading in trained model 
    with open("models/rnn_model.json", "r") as json_file:
        rnn = model_from_json(json_file.read())
    rnn.load_weights('models/rnn_20.h5')

    # Predict
    with open(outfilepath, 'w') as o:
        o.write("id,tags\n") 
        y = predict(rnn, x_test, batch_size=64) 
        for i in range(len(y)):   
            o.write(str(i))
            for j in range(len(y[i])):
                if j == 0:
                    o.write(',"'+str(y[i][j])[2:-2])
                else:
                    o.write(','+str(y[i][j])[2:-2])
            o.write('"\n')
    print("Testing result stored in %s" % outfilepath)

if __name__ == "__main__":
    nn_test(sys.argv[1], sys.argv[2]) 
