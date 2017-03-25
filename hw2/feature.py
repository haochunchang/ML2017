import numpy as np

class Xfeature(object):
    '''
    Feature of X_train
    '''
    def __init__(self, train_x, train_y):
        self.__data = train_x
        self.__label = train_y

    def __len__(self):
        '''
        Length of training data. (3xxxx)
        '''
        return self.__data.shape[0]

    def __getitem__(self, index):
        return self.__data[index, :]
   
    def get_label(self, index):
        return self.__label[index]
     
    def preprocess(self):
        '''
        Convert pandas dataframe into numpy array.
        '''
        self.__data = np.array(self.__data)
        self.__label = np.array(self.__label).flatten()
        return self     

    def __hash(self, vector):
        '''
        Input a vector of continuous feature
        Convert it to binned number.
        (e.g. 17~27: 1, 27~37: 2,...)
        '''
        vmax = int(vector.max())
        vmin = int(vector[vector > 0].min())
        bin_size = (vmax - vmin) // 5
        nbin = 5
        
        for i in range(len(vector)):
            if vector[i] < vmin+bin_size:
                vector[i] = 1
            elif (vector[i] >= vmin+bin_size and vector[i] < vmin+bin_size*2):
                vector[i] = 2
            elif vmin+bin_size <= vector[i] and vector[i]< vmin+bin_size*3:
                vector[i] = 3
            elif vmin+bin_size <= vector[i] and vector[i]< vmin+bin_size*4:
                vector[i] = 4
            else:
                vector[i] = 5

        return vector


    def bucketize(self):
        '''
        Bucketize continuous feature into catergorical feature.
        Hash each bucket
        '''
        for i in [0, 1, 3, 4, 5]:
            self.__data[:, i] = self.__hash(self.__data[:, i])

        return self

    def normalize(self):
        fmax = (self.__data).max(axis=0)
        for i in range(len(fmax)):
            if fmax[i] != 0:
                self.__data[:,i] = self.__data[:,i] / fmax[i]
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
        pre_seed = []
        seed = [i for i in np.random.randint(0, self.__data.shape[0], size=size) if i not in pre_seed]
        batch_x = self.__data[seed, :]
        batch_y = self.__label[seed]
        
        for s in seed:
            pre_seed.append(s)       
 
        return batch_x, batch_y
