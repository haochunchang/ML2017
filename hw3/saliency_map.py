import os, pickle
from keras.models import load_model
from termcolor import colored, cprint
from PIL import Image
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

def main():
    model_name = "cnn_report.h5"
    model_path = os.path.join('models/', model_name)
    emotion_classifier = load_model(model_path)
    print(colored("Loaded model from {}".format(model_name), 'yellow', attrs=['bold']))

    # Load in validation data
    with open('validation_data_augmented.pkl', 'rb') as v:
        x_test, y_test, _ = pickle.load(v)
    
    private_pixels = [ x_test[i].reshape(1, 48, 48, 1) for i in range(len(x_test)) ]
    
    input_img = emotion_classifier.input
    img_ids = [i for i in range(5)]

    for idx in img_ids:
        val_proba = emotion_classifier.predict(private_pixels[idx])
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0] * 255.0
        fn = K.function([input_img, K.learning_phase()], [grads])

        heatmap = fn([private_pixels[idx], 0])
        heatmap = heatmap[0].reshape(48, 48)
        
        thres = 0.5
        see = private_pixels[idx].reshape(48, 48)
       
        # Plot original image
        plt.figure()
        plt.imshow(see*255.0,cmap='gray')
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(ori_dir, 'original_{}.png'.format(idx)), dpi=100)

        see[np.where(heatmap <= thres)] = np.mean(see)
       
        # Plot saliency heatmap
        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(cmap_dir, 'heatmap_{}.png'.format(idx)), dpi=100)

        # Plot "Little-heat-part masked"
        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(partial_see_dir, '{}.png'.format(idx)), dpi=100)

if __name__ == "__main__":
    base_dir = os.path.join('./')
    img_dir = os.path.join(base_dir, 'image')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    cmap_dir = os.path.join(img_dir, 'cmap')
    if not os.path.exists(cmap_dir):
        os.makedirs(cmap_dir)
    partial_see_dir = os.path.join(img_dir,'partial_see')
    if not os.path.exists(partial_see_dir):
        os.makedirs(partial_see_dir)
    ori_dir = os.path.join(img_dir, 'original')
    if not os.path.exists(ori_dir):
        os.makedirs(ori_dir)
 
    main()
