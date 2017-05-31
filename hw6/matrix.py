import numpy as np
import pandas as pd
import os, sys
import keras
from keras.layers import Input, Embedding, Dense, Dropout
from keras.layers import merge, Flatten, Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
import keras.backend as K

def rmse(y_true, y_pred):
    mse = K.mean((y_true - y_pred)**2)
    return K.sqrt(mse + K.epsilon())    

def build_model(n_users, n_movies, factors=20):
        
    movie_input = Input(shape=[1])
    movie_vec = Flatten()(Embedding(n_movies, factors, input_length=1)(movie_input))
    movie_vec = Dropout(0.5)(movie_vec)

    user_input = Input(shape=[1])
    user_vec = Flatten()(Embedding(n_users, factors, input_length=1)(user_input))
    user_vec = Dropout(0.5)(user_vec)

    nn = merge([movie_vec, user_vec], mode='dot')
    
    movie_bias = Flatten()(Embedding(n_movies, 1, name='movie_bias', input_length=1)(movie_input)) 
    user_bias = Flatten()(Embedding(n_users, 1, name='users_bias', input_length=1)(user_input))

    nn = Add()([nn, movie_bias, user_bias])
    result = Dense(1, activation='relu')(nn)

    model = Model([movie_input, user_input], result)
    model.compile(optimizer='adam', loss='mean_squared_error')
    print(model.summary())   
    return model
    
def main(train_filepath):

    # Load in training data and preprocessing
    train = pd.read_csv(train_filepath)
 
    users = train['UserID'].values
    movies = train['MovieID'].values
    ratings = train['Rating'].values
 
    n_users = users.max()
    n_movies = movies.max()
    
    # Build model and fit it
    model = build_model(n_users=n_users, n_movies=n_movies, factors=256)

    model_json = model.to_json()    
    with open("models/dnn_model.json", "w") as json_file:
        json_file.write(model_json)

    callbacks = [EarlyStopping('val_loss', patience=2), \
                 ModelCheckpoint('./models/dnn.h5', save_best_only=True, verbose=1, monitor='val_loss')]
    
    model.fit([movies, users], ratings, batch_size=512, epochs=100, shuffle=True,\
              validation_split=.1, callbacks=callbacks)

    

if __name__ == "__main__":
    main(sys.argv[1])
