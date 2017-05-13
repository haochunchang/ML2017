import sys, os, pickle
import glob
import PIL
import numpy as np
from scipy import misc
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVR as SVR
from generate_data import get_eigenvalues


def predict_dim(data):
    
    # Load model
    with open('model/svr.pkl', 'rb') as f:
        model = pickle.load(f)

    # predict
    dims = []
    test = []
    evals = get_eigenvalues(data)
    test.append(evals)
    test = np.array(test)
    dims = model.predict(test)

    return np.exp(dims)

def main():
    
    # Load in hand rotation data
    filelist = sorted(glob.glob('./data/hand/resized/*.png'))
    image_list = [misc.imread(filename) for filename in filelist]   
    imgs = np.array(image_list).reshape(len(image_list), 32*34)

    # Resized Images
    ''' 
    basewidth = 32
    for i in range(len(image_list)):
        img = PIL.Image.fromarray(imgs[i])
        wpercent = (basewidth / float(imgs[i].shape[0]))
        hsize = int((float(imgs[i].shape[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        img.save('./data/hand/resized/hand_seq{}.png'.format(i+1))
    '''
    dims = predict_dim(imgs)
    print(dims.mean())
    
if __name__ == "__main__":
    main()
