import numpy as np
import sys, csv
import feature
from PIL import Image

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers 
from keras import backend as K
from keras.callbacks import TensorBoard
K.set_image_dim_ordering('tf')

def cnn_train(train_filepath, batch_size=128, epochs=10):

    num_classes = 7

    # Load in training data
    train = feature.feature(train_filepath).preprocess()
    train_norm = train.normalize()
    
    # convert class vectors to binary class matrices
    x_test, y_test = train_norm.sample_val(2000)
    x_train = train_norm.get_feature()
    y_train = keras.utils.to_categorical(train.get_label(), num_classes) 
    y_test = keras.utils.to_categorical(y_test, num_classes)
 
    # Define CNN model
    cnn = Sequential()
    cnn.add(Conv2D(16, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=(48, 48, 1), padding='same'))
    cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same' )) 
    cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn.add(Flatten())
    cnn.add(Dropout(0.25))
    cnn.add(Dense(256, activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(num_classes, activation='softmax'))

    # Compile & save model summary
    cnn.compile(loss=keras.losses.categorical_crossentropy,
                optimizer='adam',
                metrics=['accuracy'])

    # Train model
    cnn.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            validation_data=(x_test, y_test),
            callbacks=[TensorBoard(log_dir='/tmp/cnn')])

    # Serialize model weights and save them
    model_json = cnn.to_json()
    with open("models/cnn_model.json", "w") as json_file:
        json_file.write(model_json)
    cnn.save_weights('models/cnn.h5')
    print("CNN model saved.") 

if __name__ == "__main__":
    cnn_train(sys.argv[1], batch_size=128, epochs=15)

