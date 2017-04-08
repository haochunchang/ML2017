import numpy as np
import sys, csv
import feature
from PIL import Image

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator 
from keras import backend as K
from keras.callbacks import TensorBoard
K.set_image_dim_ordering('tf')

def cnn_train(train_filepath, batch_size=128, epochs=10, data_augmentation=False, pretrained=None):

    num_classes = 7

    # Load in training data
    train = feature.feature(train_filepath).preprocess()
    train_norm = train.normalize()
    
    # convert class vectors to binary class matrices
    x_test, y_test = train_norm.sample_val(3000)
    x_train = train_norm.get_feature()
    y_train = keras.utils.to_categorical(train_norm.get_label(), num_classes) 
    y_test = keras.utils.to_categorical(y_test, num_classes)
 
    # Define CNN model
    cnn = Sequential()
    cnn.add(ZeroPadding2D((1,1),input_shape=(48,48,1)))
    cnn.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    cnn.add(ZeroPadding2D((1,1))) 
    cnn.add(Conv2D(16, (3, 3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same')) 
    cnn.add(Dropout(0.25))   
 
    cnn.add(ZeroPadding2D((1,1)))
    cnn.add(Conv2D(32, (3, 3), activation='relu'))
    cnn.add(ZeroPadding2D((1,1)))
    cnn.add(Conv2D(32, (3, 3), activation='relu'))
    cnn.add(ZeroPadding2D((1,1)))
    cnn.add(Conv2D(32, (3, 3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn.add(Dropout(0.25))

    cnn.add(ZeroPadding2D((1,1)))
    cnn.add(Conv2D(64, (3, 3), activation='relu'))
    cnn.add(ZeroPadding2D((1,1)))
    cnn.add(Conv2D(64, (3, 3), activation='relu'))
    cnn.add(ZeroPadding2D((1,1)))
    cnn.add(Conv2D(64, (3, 3), activation='relu')) 
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn.add(Dropout(0.25))

    cnn.add(ZeroPadding2D((1,1)))
    cnn.add(Conv2D(128, (3, 3), activation='relu'))
    cnn.add(ZeroPadding2D((1,1)))
    cnn.add(Conv2D(128, (3, 3), activation='relu'))
    cnn.add(ZeroPadding2D((1,1)))
    cnn.add(Conv2D(128, (3, 3), activation='relu')) 
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn.add(Dropout(0.25))

    cnn.add(Flatten())
    cnn.add(Dense(1024, activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(1024, activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(num_classes, activation='softmax'))

    # Compile & print model summary
    cnn.compile(loss='categorical_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])
    print(cnn.summary())
    
    if pretrained != None:
        cnn.load_weights(pretrained)
        print('Continue Training.')    

    # Train model
    if not data_augmentation:
        print('Not using data augmentation.')
        cnn.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                shuffle=True,
                validation_data=(x_test, y_test),
                callbacks=[TensorBoard(log_dir='/tmp/cnn')])
    else:
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False) # randomly flip images

        datagen.fit(x_train)
        cnn.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                          steps_per_epoch=x_train.shape[0] // batch_size,
                          epochs=epochs, validation_data=(x_test, y_test), 
                          callbacks=[TensorBoard(log_dir='/tmp/cnn')])

    return cnn

def save_model(cnn, epochs):
        
    # Serialize model weights and save them
    model_json = cnn.to_json()
    with open("models/cnn_model.json", "w") as json_file:
        json_file.write(model_json)
    cnn.save_weights('models/cnn_'+str(epochs)+'.h5')
    print("CNN model saved.") 


def cnn_main(train_filepath):
    
    model_filepath = 'models/cnn_epoch50.h5'
    epochs = 50 
    cnn = cnn_train(train_filepath, batch_size=128, epochs=epochs, data_augmentation=False, pretrained=None)    
    save_model(cnn, epochs)

if __name__ == "__main__":
    cnn_main(sys.argv[1])

