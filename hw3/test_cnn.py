import numpy as np
import sys
import keras
from utils import feature, utils
from keras.models import model_from_json

def nn_test(test_filepath, outfilepath):
    
    # Loading in testing data & Preprocessing
    test = feature.testfeature(test_filepath).preprocess()
    test_norm = test.normalize()

    # Loading in trained model 
    with open("models/cnn_sbl.json", "r") as json_file:
        cnn = model_from_json(json_file.read())
    cnn.load_weights('models/cnn_sbl.h5')

    # Predict
    with open(outfilepath, 'w') as o:
        o.write("id,label\n") 
        y = cnn.predict(test_norm.get_feature(), batch_size=128, verbose=1) 
        y = np.argmax(y, axis=1)
        for i in range(len(test)):   
            o.write(str(i)+","+str(y[i]))
            o.write("\n")
    print("Testing result stored in %s" % outfilepath)

if __name__ == "__main__":
    nn_test(sys.argv[1], sys.argv[2]) 
