import numpy as np
import sys, csv, pickle
from utils import feature, utils
from PIL import Image

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint 
from keras import backend as K
from keras.callbacks import TensorBoard, Callback
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

def cnn_train(train_filepath, batch_size=128, epochs=10, data_augmentation=False, pretrained=None):

    num_classes = 7

    # Load in training data
    train = feature.feature(train_filepath).preprocess()
    train_norm = train.normalize()
    
    # convert class vectors to binary class matrices
    #x_test, y_test, seed = train_norm.sample_val(3000)
    #with open('validation_data_augmented.pkl', 'wb') as v:
    #    pickle.dump((x_test, y_test, seed), v)
    
    with open('validation_data_augmented.pkl', 'rb') as v:
        x_test, y_test, seed = pickle.load(v)
    
    x_train = train_norm.delete(seed)
    x_train = train_norm.get_feature()
    y_train = keras.utils.to_categorical(train_norm.get_label(), num_classes) 
    y_test = keras.utils.to_categorical(y_test, num_classes)
 
    # Define CNN model
    cnn = Sequential()
    cnn.add(Conv2D(64, 3, input_shape=(48,48,1), padding='valid'))
    cnn.add(LeakyReLU(alpha=.001))
    cnn.add(BatchNormalization())
    cnn.add(ZeroPadding2D((1,1)))
    cnn.add(Conv2D(64, 3, padding='valid'))
    cnn.add(LeakyReLU(alpha=.001))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(2, strides=2)) 
    cnn.add(Dropout(0.25))

    cnn.add(ZeroPadding2D((1,1)))
    cnn.add(Conv2D(128, 3, padding='valid'))
    cnn.add(LeakyReLU(alpha=.001))
    cnn.add(BatchNormalization())
    cnn.add(ZeroPadding2D((1,1)))
    cnn.add(Conv2D(128, 3, padding='valid'))
    cnn.add(LeakyReLU(alpha=.001))
    cnn.add(BatchNormalization()) 
    cnn.add(MaxPooling2D(2, strides=2)) 
    cnn.add(Dropout(0.25)) 

    cnn.add(ZeroPadding2D((1,1)))
    cnn.add(Conv2D(256, 3, padding='valid'))
    cnn.add(LeakyReLU(alpha=.001))
    cnn.add(BatchNormalization())
    cnn.add(ZeroPadding2D((1,1)))
    cnn.add(Conv2D(256, 3, padding='valid'))
    cnn.add(LeakyReLU(alpha=.001))
    cnn.add(BatchNormalization()) 
    cnn.add(MaxPooling2D(2, strides=2)) 
    cnn.add(Dropout(0.25)) 
    
    cnn.add(Flatten())
    cnn.add(Dropout(0.25))
    cnn.add(Dense(128))
    cnn.add(LeakyReLU(alpha=.001))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.25))
    cnn.add(Dense(128))
    cnn.add(LeakyReLU(alpha=.001))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.25)) 
    cnn.add(Dense(num_classes, activation='softmax'))

    # Compile & print model summary
    cnn.compile(loss='categorical_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])
    print(cnn.summary())
   
    model_json = cnn.to_json()
    with open("models/cnn_model.json", "w") as json_file:
        json_file.write(model_json)
 
    if pretrained != None:
        cnn.load_weights(pretrained)
        print('Continue Training.')    

    # Train model
    print('Using real-time data augmentation.')
    history = History()
    checkpointer = ModelCheckpoint(filepath="./checkpoints/weights.{epoch:02d}-{val_acc:.2f}.h5", verbose=1, save_best_only=True, monitor='val_acc')  
    datagen = ImageDataGenerator(
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True)  # randomly flip images

    datagen.fit(x_train)
    cnn.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                      steps_per_epoch=x_train.shape[0] // batch_size,
                      epochs=epochs, validation_data=(x_test, y_test),
                      callbacks=[TensorBoard(log_dir='./log/events.epochs'+str(epochs)), checkpointer, history])

    utils.dump_history('./log/',history)
    
    # Serialize model weights and save them
    cnn.save_weights('models/cnn_'+str(epochs)+'.h5')
    print("CNN model saved.") 
        
def cnn_main(train_filepath):
    
    model_filepath = None
    epochs = 80
    cnn_train(train_filepath, batch_size=64, epochs=epochs, data_augmentation=True, pretrained=model_filepath)    

if __name__ == "__main__":
    cnn_main(sys.argv[1])

