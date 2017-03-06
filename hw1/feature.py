import numpy as np
import random, math
import pandas as pd

class Feature(object):
    '''
    List of features. Each entry is a matrix of a single day.
    '''
    def __init__(self, data):
        self.__data, self.__label, self.mu, self.std = self._preprocess(data)
    
    def _preprocess(self, train):
        feature_lst = []
        label = []
        train = train.drop(['測站', '測項', '日期'], axis=1)
        train = np.array(train.replace('NR', 0), dtype=np.float32)
        
        train_norm = np.zeros((18,1))
        for i in range(240):
            single_day = train[i*18:(i+1)*18, :]
            train_norm = np.append(train_norm, single_day, axis=1)
        
        train_norm = np.delete(train_norm, 0, axis=1)    
        pm25 = train_norm[9, :]
        self.labelmu = np.array(pm25).mean()
        self.labelstd = np.array(pm25).std()
        
        # Normalization
        mu = train_norm.mean(axis=1)
        std = train_norm.std(axis=1)
        for j in range(train_norm.shape[0]):
            if std[j] != 0:
                train_norm[j,] = (train_norm[j,] - mu[j]) / std[j]
        
        for mon in range(12):
            for hr in range(471):
                feature = np.insert(train_norm[:, (mon*480)+hr:(mon*480)+hr+9], 0, 1, axis=0)
                feature_lst.append(feature)
                label.append(train_norm[9, (mon*480)+hr+9])
        return feature_lst, label, mu, std
    
    def shuffle(self):
        '''    
        Shuffle feature order
        '''
        fea_lab = list(zip(self.__data, self.__label))
        random.shuffle(fea_lab)
        feature_lst, label = zip(*fea_lab)
        self.__data = list(feature_lst)
        self.__label = list(label)

        return self
 
    def flatten(self):
        '''
        Flatten feature and add bias
        return shape: (163,)
        '''
        for i in range(len(self.__data)):
            self.__data[i] = np.insert(self.__data[i].flatten(), 0, 1)
    
    def split(self, start, stop):
        self.__data = self.__data[start:stop]
        self.__label = self.__label[start:stop]
        return self
    
    def get_label_std(self):
        return self.labelmu   
    
    def get_label_mu(self):
        return self.labelstd
     
    def get_allfeature(self):
        return self.__data

    def get_alllabel(self):
        return self.__label
    
    def sample_val(self, size):
        '''
        Random sample validation set of given size from self.
        '''
        seed = random.sample(range(len(self.__data)), size)
        not_seed = [i for i in range(len(self.__data)) if i not in seed]
        
        val_f = [self.__data[i] for i in seed]
        val_l = [self.__label[i] for i in seed]
        self.__data = [self.__data[i] for i in not_seed]
        self.__label = [self.__label[i] for i in not_seed]
    
        return val_f, val_l

    def get_label(self, index):
        return self.__label[index]

    def __getitem__(self, index):
        return self.__data[index]
    
    def __len__(self):
        return len(self.__data)

class TestFeature(object):
    '''
    Testing feature list with different preprocessing.
    '''
    def __init__(self, data, mu, std):
        self.__data, self.__label, self.__mu, self.__std = self._preprocess(data, mu, std)

    def _preprocess(self, test, mu, std):
        feature_lst = []
        label = []
        test = test.iloc[:, 2:].replace('NR', 0)
        test = np.array(test, dtype=np.float32)
        test_norm = np.zeros((18, 1))
    
        for i in range(240):
            test_norm = np.append(test_norm, test[i*18:i*18+18, :], axis=1) 
         
        test_norm = np.delete(test_norm, 0, 1)
        
        # Normalize with training mu and std
        for j in range(test_norm.shape[0]):
            if std[j] != 0:
                test_norm[j,] = (test_norm[j,] - mu[j]) / std[j] 
    
        for i in range(240):
            feature_lst.append(np.insert(test_norm[:, i*9:i*9+9], 0, 1, axis=0))

        return feature_lst, label, mu, std
    
    def __getitem__(self, index):
        return self.__data[index]
    
    def __len__(self):
        return len(self.__data)
    
    def flatten(self):
        '''
        Flatten feature and add bias
        return shape: (163,)
        '''
        for i in range(len(self.__data)):
            self.__data[i] = np.insert(self.__data[i].flatten(), 0, 1)

