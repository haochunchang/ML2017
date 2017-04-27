import numpy as np
import sys, csv, pickle, random
from utils import feature, utils

import keras
from keras.utils import plot_model
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
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

def semi_train(train_filepath, batch_size=64, epochs=80):
    
    num_classes = 7

    # Load in training data
    train = feature.feature(train_filepath).preprocess()
    train_norm = train.normalize()
        
    # Loading in validation data
    with open('validation_data_augmented.pkl', 'rb') as v:
        x_val, _, seed = pickle.load(v)
    '''
    # Loading in trained model 
    with open("models/cnn_model.json", "r") as json_file:
        pre_cnn = model_from_json(json_file.read())
    pre_cnn.load_weights('models/cnn_report.h5')

    # Predict validation data
    prediction = pre_cnn.predict(x_val, batch_size=64, verbose=1) 
    y_prediction = np.argmax(prediction, axis=1)
    y_prediction = keras.utils.to_categorical(y_prediction, 7)
    with open('semi_unlabel_data_predicted.pkl', 'wb') as v:
        pickle.dump(y_prediction, v)
    
    print("Validation result predicted")
    '''
    with open('semi_unlabel_data_predicted.pkl', 'rb') as v:
        y_prediction = pickle.load(v)
    
    x_train = train_norm.delete(seed)
    x_train = train_norm.get_feature()
    x_train = np.concatenate((x_train, x_val), axis=0)

    y_train = keras.utils.to_categorical(train_norm.get_label(), num_classes) 
    y_train = np.concatenate((y_train, y_prediction), axis=0)
 
    # Sample other validation data
    seed2 = random.sample(range(x_train.shape[0]), 3000) 
    x_test = x_train[seed2]
    y_test = y_train[seed2]
    np.delete(x_train, seed2, axis=0)
    np.delete(y_train, seed2, axis=0) 

    # Define CNN model
    cnn = Sequential()
    cnn.add(Conv2D(64, 3, input_shape=(48,48,1), padding='valid', activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(ZeroPadding2D((1,1)))
    cnn.add(Conv2D(64, 3, padding='valid', activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(2, strides=2)) 
    cnn.add(Dropout(0.25))

    cnn.add(ZeroPadding2D((1,1)))
    cnn.add(Conv2D(128, 3, padding='valid', activation='relu'))
    cnn.add(ZeroPadding2D((1,1)))
    cnn.add(Conv2D(128, 3, padding='valid', activation='relu'))
    cnn.add(BatchNormalization()) 
    cnn.add(MaxPooling2D(2, strides=2)) 
    cnn.add(Dropout(0.25)) 

    cnn.add(ZeroPadding2D((1,1)))
    cnn.add(Conv2D(256, 3, padding='valid', activation='relu'))
    cnn.add(BatchNormalization()) 
    cnn.add(MaxPooling2D(2, strides=2)) 
    cnn.add(Dropout(0.25)) 
    
    cnn.add(Flatten())
    cnn.add(Dropout(0.25))
    cnn.add(Dense(128, activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.25))
    cnn.add(Dense(128, activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.25)) 
    cnn.add(Dense(num_classes, activation='softmax'))

    # Compile & print model summary
    cnn.compile(loss='categorical_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])
    print(cnn.summary())
    
    model_json = cnn.to_json()
    with open("models/semi_model.json", "w") as json_file:
        json_file.write(model_json)
    
    plot_model(cnn, to_file='semi_cnn.png', show_shapes=True) 
 
    # Train model
    print('Using real-time data augmentation.')
    history = History()
    checkpointer = ModelCheckpoint(filepath="./checkpoints/semi/weights.{epoch:02d}-{val_acc:.2f}.h5", verbose=1, save_best_only=True, monitor='val_acc')  
    datagen = ImageDataGenerator(
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True)  # randomly flip images

    datagen.fit(x_train)
    cnn.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                      steps_per_epoch=x_train.shape[0] // batch_size,
                      epochs=epochs, validation_data=(x_test, y_test),
                      callbacks=[TensorBoard(log_dir='./log/semi/events.epochs'+str(epochs)), checkpointer, history])

    utils.dump_history('./log/semi/',history)
    
    # Serialize model weights and save them
    cnn.save_weights('models/semi_'+str(epochs)+'.h5')
    print("semi-CNN model saved.") 

if __name__ == "__main__":
    semi_train(sys.argv[1]) 
