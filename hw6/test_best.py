import numpy as np
import pandas as pd
import os, sys, pickle
import keras
from keras.models import model_from_json
from keras.utils import plot_model
   
def main(test_dir):

    # Load in training data and preprocessing
    test = pd.read_csv(os.path.join(test_dir, 'test.csv'))
    
    users = test['UserID'].values
    movies = test['MovieID'].values

    # Load models
    with open("models/sbl_model.json", "r") as json_file:
        dnn = model_from_json(json_file.read())
    
    #plot_model(dnn, to_file='bias_model.png', show_shapes=True)
 
    dnn.load_weights('models/sbl.h5')
    
    pred = dnn.predict([movies, users], batch_size=256, verbose=1) 
    #sub = pd.DataFrame()
    #sub['TestDataID'] = test['TestDataID']
    #sub['Rating'] = pred
    #sub.to_csv(outfilepath, index=False) 
    #print('Test result stored in {}'.format(outfilepath))
    return pred
 
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
