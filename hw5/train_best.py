import numpy as np
import sys, pickle
import utils

import keras
from keras.utils import plot_model
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.layers import LSTM, GRU, BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping 
from keras import backend as K
from keras.callbacks import TensorBoard, Callback
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

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

def f1_score(y_true, y_pred):
    thresh = 0.4
    
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))



def rnn_train(x_train, y_train, batch_size=128, epochs=10, pretrained=None):
       
    with open('text_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    x_train = tokenizer.sequences_to_matrix(x_train, mode='tfidf')

    # Split validation data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=6)
 
    word_index = tokenizer.word_index
    emb = utils.load_embedding(100, pretrain='glove')    

    # Define RNN model
    rnn = Sequential()
    #rnn.add(Embedding(len(word_index) + 1, 100, weights=[emb],
    #                        input_length=x_train.shape[1],
    #                        trainable=False))
    #rnn.add(Flatten()) 
    #rnn.add(GRU(128, activation='tanh', dropout=0.1))
    rnn.add(Dense(256, input_shape=((x_train.shape[1], )), activation='relu'))
    rnn.add(Dropout(0.2))
    rnn.add(Dense(128, activation='relu'))
    rnn.add(Dropout(0.2))
    rnn.add(Dense(128, activation='relu'))
    rnn.add(Dropout(0.2))
    rnn.add(Dense(64,activation='relu'))
    rnn.add(Dropout(0.2))
    rnn.add(Dense(64,activation='relu'))
    rnn.add(Dropout(0.2))
    rnn.add(Dense(y_train.shape[1], activation='sigmoid'))
    
    # Compile & print model summary
    rnn.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=[f1_score])
    print(rnn.summary())
   
    model_json = rnn.to_json()
    with open("models/rnn_model.json", "w") as json_file:
        json_file.write(model_json)
    
    plot_model(rnn, to_file='rnn_model.png', show_shapes=True)
    
    #history = History()
    checkpointer = ModelCheckpoint(filepath="./models/rnn.h5", 
                    verbose=1, save_best_only=True, monitor='val_f1_score', mode='max')  
    earlystopping = EarlyStopping(monitor='val_f1_score', patience = 10, verbose=1, mode='max')
    if pretrained != None:
        rnn.load_weights(pretrained)
        print('Continue Training.')    
        rnn.fit(x_train, y_train, batch_size=batch_size, initial_epoch=20,
            verbose=1, epochs=epochs, validation_data=(x_val, y_val),
            callbacks=[earlystopping, checkpointer])
 
    else:
        # Train model
        rnn.fit(x_train, y_train, batch_size=batch_size,
            verbose=1, epochs=epochs, validation_data=(x_val, y_val),
            callbacks=[earlystopping, checkpointer])

    

def rnn_main(train_filepath):
    
    #x_train = np.loadtxt('x_train_tfidf.txt')
    model_filepath = None#'models/rnn.h5'
    epochs = 100
    x_train, y_train = utils.preprocess(train_filepath, './data/test_data.csv', save_token=False)
    rnn_train(x_train, y_train, batch_size=128, epochs=epochs, pretrained=model_filepath)    

if __name__ == "__main__":
    rnn_main(sys.argv[1])

