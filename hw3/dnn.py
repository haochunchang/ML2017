import numpy as np
import sys, csv, pickle
from utils import feature, utils

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint 
from keras import backend as K
from keras.callbacks import TensorBoard, Callback
from keras.utils import plot_model

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


def dnn_train(train_filepath, batch_size=128, epochs=10, data_augmentation=False, pretrained=None):

    num_classes = 7

    # Load in training data
    train = feature.feature(train_filepath).preprocess()
    train_norm = train.normalize()
    
    # convert class vectors to binary class matrices
    x_test, y_test, seed = train_norm.sample_val(3000)
    with open('DNN_validation_data.pkl', 'wb') as v:
        pickle.dump((x_test, y_test, seed), v)
    #with open('dnn/DNN_validation_data.pkl', 'rb') as v:
    #    x_test, y_test, seed = pickle.load(v)
    
    #x_train = train_norm.delete(seed)
    x_train = train_norm.get_feature()
    y_train = keras.utils.to_categorical(train_norm.get_label(), num_classes) 
    y_test = keras.utils.to_categorical(y_test, num_classes)
 
    # Define CNN model
    dnn = Sequential()
    dnn.add(Flatten(input_shape=(48,48,1)))
    dnn.add(Dense(512, input_shape=(2304,)))
    dnn.add(LeakyReLU(alpha=.001))
    dnn.add(BatchNormalization())
    dnn.add(Dense(512))
    dnn.add(LeakyReLU(alpha=.001))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(0.25))
    dnn.add(Dense(512))
    dnn.add(LeakyReLU(alpha=.001))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(0.25))
    dnn.add(Dense(256))
    dnn.add(LeakyReLU(alpha=.001))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(0.25))
    dnn.add(Dense(256))
    dnn.add(LeakyReLU(alpha=.001))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(0.25)) 
    dnn.add(Dense(128))
    dnn.add(LeakyReLU(alpha=.001))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(0.25))
    dnn.add(Dense(128))
    dnn.add(LeakyReLU(alpha=.001))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(0.25))
    dnn.add(Dense(128))
    dnn.add(LeakyReLU(alpha=.001))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(0.25)) 
    dnn.add(Dense(128))
    dnn.add(LeakyReLU(alpha=.001))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(0.25))  
    dnn.add(Dense(num_classes, activation='softmax'))

    # Compile & print model summary
    dnn.compile(loss='categorical_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])
    print(dnn.summary())
  
    plot_model(dnn, to_file='dnn_report_model.png', show_shapes=True)
 
    model_json = dnn.to_json()
    with open("models/dnn_model.json", "w") as json_file:
        json_file.write(model_json)
 
    if pretrained != None:
        dnn.load_weights(pretrained)
        print('Continue Training.')    

    # Train model
    print('Using real-time data augmentation.')
    history = History()
    checkpointer = ModelCheckpoint(filepath="./checkpoints/dnn/weights.{epoch:02d}-{val_acc:.2f}.h5", verbose=1, save_best_only=True, monitor='val_acc')  
    datagen = ImageDataGenerator(
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True)  # randomly flip images

    datagen.fit(x_train)
    dnn.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                          steps_per_epoch=x_train.shape[0] // batch_size,
                          epochs=epochs, validation_data=(x_test, y_test),
                          callbacks=[TensorBoard(log_dir='./log/dnn/events.epochs'+str(epochs)+'augmented'), checkpointer, history])

    utils.dump_history('./log/dnn/',history)
        
    # Serialize model weights and save them
    dnn.save_weights('models/dnn_'+str(epochs)+'.h5')
    
    print("DNN model saved.") 

def dnn_main(train_filepath):
    
    model_filepath = None
    epochs = 80
    dnn_train(train_filepath, batch_size=64, epochs=epochs, data_augmentation=True, pretrained=model_filepath)    

if __name__ == "__main__":
    dnn_main(sys.argv[1])

