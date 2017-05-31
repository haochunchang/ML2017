import numpy as np
import pandas as pd
import os, sys, pickle
import keras
from keras.models import model_from_json
from keras.preprocessing import text, sequence

def main(data_dir, outfilepath):

    # Load in training data and preprocessing
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    movie_other = pd.read_csv(os.path.join(data_dir, 'movies.csv'), sep='::', engine='python')
    user_other = pd.read_csv(os.path.join(data_dir, 'users.csv'), sep='::', engine='python')
    movie_other = movie_other.rename(columns={'movieID': 'MovieID'})
    
    # Preprocess users data
    user_other['Gender'] = [1 if i == 'M' else 0 for i in user_other['Gender']]
    user_other = user_other.drop(['Zip-code'], axis=1)

    test = pd.merge_ordered(test, user_other, on='UserID', how='inner')
    test = pd.merge_ordered(test, movie_other, on='MovieID', how='inner')
    test_other = test.as_matrix(columns=['Gender','Age','Occupation'])
    
    # Preprocess movies data    
    with open('title_tokenizer.pkl', 'rb') as f:
        title_tokenizer = pickle.load(f)
    with open('genre_tokenizer.pkl', 'rb') as f:
        genre_tokenizer = pickle.load(f)

    title = title_tokenizer.texts_to_sequences(test.Title)
    genre = genre_tokenizer.texts_to_sequences(test.Genres)
    title = sequence.pad_sequences(title)
    genre = sequence.pad_sequences(genre)
 
    users = test['UserID'].values
    movies = test['MovieID'].values

    # Load models
    with open("models/matplus_model.json", "r") as json_file:
        dnn = model_from_json(json_file.read())
    dnn.load_weights('models/matplus_dnn.h5')

    pred = dnn.predict([movies, users, title, genre, test_other], batch_size=512, verbose=1) 
    sub = pd.DataFrame()
    sub['TestDataID'] = test['TestDataID']
    sub['Rating'] = pred
    sub = sub.sort_values('TestDataID', axis=0)
    sub.to_csv(outfilepath, index=False) 
    print('Test result stored in {}'.format(outfilepath))
 
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
