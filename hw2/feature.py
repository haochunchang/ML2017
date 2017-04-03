import numpy as np
import random

class Xfeature(object):
    '''
    Feature of X_train
    '''
    def __init__(self, train_x, train_y):
        self.__data = train_x
        self.__label = train_y
        self.sample_times = 0

    def __len__(self):
        '''
        Length of training data. (3xxxx)
        '''
        return self.__data.shape[0]

    def __getitem__(self, index):
        return self.__data[index, :]
   
    def get_label(self, index):
        return self.__label[index]
    
    def get(self):
        return self.__data, self.__label        
 
    def preprocess(self):
        '''
        Convert pandas dataframe into numpy array.
        '''
        self.__data = np.array(self.__data)
        self.__label = np.array(self.__label).flatten()
        return self     

    def __hash(self, vector, nbin):
        '''
        Input a vector of continuous feature
        Convert it to binned number.
        (e.g. 17~27: 1, 27~37: 2,...)
        '''
        vmax = int(vector.max())
        vmin = int(vector[vector > 0].min())
        bin_size = (vmax - vmin) // nbin
                
        for i in range(len(vector)):
            if vector[i] < vmin:
                vector[i] = 1
                continue
            elif vector[i] > vmin+bin_size*(nbin-1):
                vector[i] = nbin
                continue
            for j in range(1, nbin):
                low = vmin+bin_size*(j-1)
                high = vmin+bin_size*j
                if low <= vector[i] and vector[i] < high:
                    vector[i] = j                
        return vector
    
    def sample_val(self, size):
        '''
        Random sample validation set of given size from self.
        '''
        seed = random.sample(range(self.__data.shape[0]), size)
        not_seed = [i for i in range(self.__data.shape[0]) if i not in seed]
        
        val_f = np.array([self.__data[i, :] for i in seed])
        val_l = np.array([self.__label[i] for i in seed])
        self.__data = np.array([self.__data[i, :] for i in not_seed])
        self.__label = np.array([self.__label[i] for i in not_seed])
        return val_f, val_l

    def bucketize(self, index, nbin):
        '''
        Bucketize continuous feature into catergorical feature.
        Hash each bucket
        '''
        self.__data[:, index] = self.__hash(self.__data[:, index], nbin)

        return self

    def cross(self, i, j):
        self.__data = np.insert(self.__data, self.__data.shape[1]-1, self.__data[:, i] * self.__data[:, j], axis=1)
        return self

    def delete(self, lst):
        new = np.delete(self.__data, lst, axis=1)
        self.__data = new
        return self

    def normalize(self):
        '''
        Standardization of continuous features.
        '''
        mu = (self.__data).mean(axis=0)
        std = self.__data.std(axis=0)
        for i in [0, 1, 3, 4, 5]:
            if std[i] != 0:
                self.__data[:,i] = self.__data[:,i] - mu[i] / std[i]
        return self
  
    def add_bias(self):
        '''
        Add bias to each feature vector.
        '''
        self.__data = np.insert(self.__data, 0, 1, axis=1)
    
    def shuffle(self):
        '''    
        Shuffle feature order
        '''
        data = np.insert(self.__data, 0, self.__label, axis=1)
        np.random.shuffle(data)
        self.__data = data[:, 1:]
        self.__label = data[:, 0]

        return self

    def sample(self, size):
        '''
        Randomly sample without replacement.
        ''' 
        seed = [i for i in range(self.sample_times*size, (self.sample_times+1)*size)]
        batch_x = self.__data[seed, :]
        batch_y = self.__label[seed]
               
        # Record sampled time          
        self.sample_times += 1       
 
        return batch_x, batch_y
