from keras.models import load_model
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os, pickle, itertools

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    model_path = os.path.join('models', 'cnn_report.h5')
    emotion_classifier = load_model(model_path)
    np.set_printoptions(precision=2)

    with open("validation_data_augmented.pkl", 'rb') as v:
        x_val, y_val, _ = pickle.load(v)
    
    predictions = emotion_classifier.predict_classes(x_val)
    y_val = np.array(y_val, dtype=np.int32).flatten()
    conf_mat = confusion_matrix(y_val, predictions)

    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
    plt.show()

if __name__ == "__main__":
    main()
