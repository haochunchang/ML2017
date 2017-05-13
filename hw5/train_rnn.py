import numpy as np
import sys, pickle
import utils

import keras
from keras.utils import plot_model
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence, text
from keras.callbacks import ModelCheckpoint 
from keras import backend as K
from keras.callbacks import TensorBoard, Callback
from sklearn.metrics import matthews_corrcoef
K.set_image_dim_ordering('tf')

class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_losses=[]
        self.val_losses=[]
        self.tr_accs=[]
        self.val_accs=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.tr_accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))

def best_threshold(model_path):

    # Loading in trained model 
    with open("models/rnn_model.json", "r") as json_file:
        rnn = model_from_json(json_file.read())
    rnn.load_weights(model_path)
  
    with open('val_data.pkl', 'rb') as f:
        x_test, y_test = pickle.load(f)

    out = rnn.predict_proba(x_test)
    out = np.array(out)
    threshold = np.arange(0.1,0.9,0.1)

    acc = []
    accuracies = []
    best_threshold = np.zeros(out.shape[1])
    for i in range(out.shape[1]):
        y_prob = np.array(out[:,i])
        for j in threshold:
            y_pred = [1 if prob>=j else 0 for prob in y_prob]
            acc.append( matthews_corrcoef(y_test[:,i],y_pred))
        acc   = np.array(acc)
        index = np.where(acc==acc.max()) 
        accuracies.append(acc.max()) 
        best_threshold[i] = threshold[index[0][0]]
        acc = []
    np.savetxt('best_threshold.txt', best_threshold)

def preprocess(train_filepath):
    
    # Load in training data
    idx = []
    tags = []
    para = []
    with open(train_filepath, 'r', encoding='latin-1') as f:
        for line in f:
            if line.split('"')[0] == 'id,tags,text\n':
                continue
            else:
                line = line.split('"')
                ix = int(line[0][:-1])
                tag = line[1]
                para_text = ''.join(line[2:])[1:]
                idx.append(ix)
                tags.append(tag)
                para.append(para_text)

    x_tokenizer = text.Tokenizer()
    x_tokenizer.fit_on_texts(para)
    seqs = x_tokenizer.texts_to_sequences(para)    

    x_train = sequence.pad_sequences(seqs)
   
    # Preproces tags into categorical label 
    y_token = text.Tokenizer(filters='', lower=False, split=',')
    y_token.fit_on_texts(tags)
    with open('tag_tokenizer.pkl', 'wb') as f:
        pickle.dump(y_token, f)
 
    y_train = np.zeros((len(tags), 38))
    tmp = 0
    for tag in tags:
        tag = tag.split(',')
        class_idx = [y_token.word_index[t]-1 for t in tag]
        for idx in class_idx:
            y_train[tmp, idx] = 1
        tmp += 1
    y_train = np.array(y_train)

    return x_train, y_train

def rnn_train(x_train, y_train, batch_size=128, epochs=10, pretrained=None):

    # Split validation data
    portion = len(x_train) // 10
    x_val = x_train[:portion]
    x_train = x_train[portion:]
    y_val = y_train[:portion]
    y_train = y_train[portion:]
    
    with open('val_data.pkl', 'wb') as f:
        pickle.dump((x_val, y_val), f)
 
    # Define RNN model
    rnn = Sequential()
    embedding_vecor_length = 32
    rnn.add(Embedding(x_train.shape[0], embedding_vecor_length))
    rnn.add(LSTM(100))
    rnn.add(Dense(y_train.shape[1], activation='sigmoid'))
    
    # Compile & print model summary
    rnn.compile(loss='binary_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])
    print(rnn.summary())
   
    model_json = rnn.to_json()
    with open("models/rnn_model.json", "w") as json_file:
        json_file.write(model_json)
    
    plot_model(rnn, to_file='rnn_model.png', show_shapes=True)
    
    if pretrained != None:
        rnn.load_weights(pretrained)
        print('Continue Training.')    
    
    # Train model
    history = History()
    checkpointer = ModelCheckpoint(filepath="./checkpoints/weights.{epoch:02d}-{val_acc:.2f}.h5", verbose=1, save_best_only=True, monitor='val_acc')  
    rnn.fit(x_train, y_train, batch_size=batch_size,
            verbose=1, epochs=epochs, validation_data=(x_val, y_val),
            callbacks=[TensorBoard(log_dir='./log/events.epochs'+str(epochs)), checkpointer, history])

    utils.dump_history('./log/',history)
    
    # Serialize model weights and save them
    rnn.save_weights('models/rnn.h5')
    print("RNN model saved.") 
        

def rnn_main(train_filepath):
    
    model_filepath = None
    epochs = 50
    x_train, y_train = preprocess(train_filepath)
    rnn_train(x_train, y_train, batch_size=64, epochs=epochs, pretrained=model_filepath)    
    best_threshold(model_path='models/rnn.h5')

if __name__ == "__main__":
    rnn_main(sys.argv[1])

