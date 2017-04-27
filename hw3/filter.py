#!/usr/bin/env python
# -- coding: utf-8 --

import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from utils import feature, utils
import numpy as np
K.set_learning_phase(0)

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func):
   
    for i in range(num_step):
        loss_value, grads_value = iter_func([input_image_data])
        input_image_data += grads_value * 1e-2
    
    filter_images = (input_image_data.reshape(48, 48), loss_value)

    return filter_images

def main():
    filter_dir = './image'
    store_path = 'filters'
    emotion_classifier = load_model('./models/cnn_report.h5')
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers)
    input_img = emotion_classifier.input

    name_ls = ['conv2d_1', 'conv2d_2']
    collect_layers = [ layer_dict[name].output for name in name_ls ]

    NUM_STEPS = 100
    RECORD_FREQ = 10 
    num_step = 100
    nb_filter = 64

    for cnt, c in enumerate(collect_layers):
        filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
        for it in range(NUM_STEPS//RECORD_FREQ):
            for filter_idx in range(nb_filter):
                input_img_data = np.random.random((1, 48, 48, 1)) # random noise
                target = K.mean(c[:, :, :, filter_idx])
                grads = normalize(K.gradients(target, input_img)[0])
                iterate = K.function([input_img], [target, grads])

                filter_imgs[it].append(grad_ascent(num_step, input_img_data, iterate))
        
        for it in range(NUM_STEPS//RECORD_FREQ):
            fig = plt.figure(figsize=(14, 8))
            for i in range(nb_filter):
                ax = fig.add_subplot(nb_filter/16, 16, i+1)
                ax.imshow(filter_imgs[it][i][0], cmap='BuGn')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.xlabel('{:.3f}'.format(filter_imgs[it][i][1]))
                plt.tight_layout()
            fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], it*RECORD_FREQ))
            img_path = os.path.join(filter_dir, '{}-{}'.format(store_path, name_ls[cnt]))
            if not os.path.isdir(img_path):
                os.mkdir(img_path)
            fig.savefig(os.path.join(img_path,'e{}'.format(it*RECORD_FREQ)))

if __name__ == "__main__":
    main()
'''
def main():
    fig = plt.figure(figsize=(14,8)) # 大小可自行決定
    for i in range(nb_filter): # 畫出每一個filter
        ax = fig.add_subplot(nb_filter/16,16,i+1) # 每16個小圖一行
        ax.imshow(image,cmap='BuGn') # image為某個filter的output或最能activate某個filter的input image
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.xlabel('whatever subfigure title you want') # 如果你想在子圖下加小標的話
        plt.tight_layout()
    fig.suptitle('Whatever title you want')
    fig.savefig(os.path.join(img_path,'Whatever filename you want')) #將圖片儲存至disk
'''
