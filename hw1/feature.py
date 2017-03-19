import numpy as np
import random, math
import pandas as pd

class Feature(object):
    '''
    List of features. Each entry is a matrix of a single day.
    '''
    def __init__(self, data):
        self.__data, self.__label, self.mu, self.std = self._preprocess(data)
        self.raw = data

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
    
    def scaling(self):
        
        data = self.raw
        train = data.drop(['測站', '測項', '日期'], axis=1)
        train = np.array(train.replace('NR', 0), dtype=np.float32)
       
        # Transform to 18 factors x #train_data 
        train_norm = np.zeros((18,1))
        for i in range(240):
            single_day = train[i*18:(i+1)*18, :]
            train_norm = np.append(train_norm, single_day, axis=1)
        train_norm = np.delete(train_norm, 0, axis=1) 
        
        # Insert pm2.5, pm10 square and pm2.5 ^ 3 
        train_norm = np.insert(train_norm, len(train_norm), train_norm[9, :] ** 2, axis=0)
        train_norm = np.insert(train_norm, len(train_norm), train_norm[8, :] ** 2, axis=0)
        train_norm = np.insert(train_norm, len(train_norm), train_norm[8, :] * train_norm[9, :], axis=0)
 
        # Standardization
        self.mu = train_norm.mean(axis=1)
        self.std = train_norm.std(axis=1)
        for j in range(train_norm.shape[0]):
            if self.std[j] != 0:
                train_norm[j,] = (train_norm[j,] - self.mu[j]) / self.std[j]
        
        # Extract standardized labels
        self.__label = [] 
        for mon in range(12):
            for hr in range(471):
                self.__label.append(train_norm[9, (mon*480)+hr+9])
        
        # Shrink unrelevent feature into 0.0 by PCC to pm2.5
        cor_mat = np.corrcoef(train_norm)[9, :]
        for i in range(len(cor_mat)):
            if abs(cor_mat[i]) < 0.2:
                train_norm[i, :] = train_norm[i, :] * 0.0
         
        # Extract features
        self.__data = []
        for mon in range(12):
            for hr in range(471):
                feature = train_norm[:, (mon*480)+hr:(mon*480)+hr+9]
                self.__data.append(feature)
       
        return self

    def add_bias(self):
        '''
        flatten feature and add bias
        Also add 9th hr pm2.5 ** 2 and 9th hr pm10 ** 2
        '''
        for i in range(len(self.__data)):
            pm25 = self.__data[i][9, 8] ** 3
            #self.__data[i] = np.insert(self.__data[i], len(self.__data[i]), pm25) 
            self.__data[i] = np.insert(self.__data[i], 0, 1)
    
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
 
    def bagging(self, other):
        '''
        Sample training data with replacement.
        '''
        new_f = []
        new_l = []
        new_f2 = []
        new_l2 = []
        for i in range(len(self.__data)):
            x = random.randint(0, len(self.__data) - 1)
            new_f.append(self.__data[x])
            new_l.append(self.__label[x])
            new_f2.append(other.__data[x])
            new_l2.append(other.__label[x])
        self.__data = new_f
        self.__label = new_l
        other.__data = new_f2
        other.__label = new_l2
        return self, other    

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
        self.raw = data

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
    
    def scaling(self):
        '''
        Scale feature value by correlation coefficient of pm2.5
        Original * 100 * abs(PCC)
        '''
        test = self.raw
        test = test.iloc[:, 2:].replace('NR', 0)
        test = np.array(test, dtype=np.float32)
        test_norm = np.zeros((18, 1))
    
        for i in range(240):
            test_norm = np.append(test_norm, test[i*18:i*18+18, :], axis=1) 
         
        test_norm = np.delete(test_norm, 0, 1)
        
        # Insert pm2.5 and pm10 square term 
        test_norm = np.insert(test_norm, len(test_norm), test_norm[9, :] ** 2, axis=0)
        test_norm = np.insert(test_norm, len(test_norm), test_norm[8, :] ** 2, axis=0)
        test_norm = np.insert(test_norm, len(test_norm), test_norm[8, :] * test_norm[9, :], axis=0)
 

        # Normalization
        for j in range(test_norm.shape[0]):
            if self.__std[j] != 0:
                test_norm[j,] = (test_norm[j,] - self.__mu[j]) / self.__std[j]
        
        cor_mat = np.corrcoef(test_norm)[9, :]
        for i in range(len(cor_mat)):
            if abs(cor_mat[i]) < 0.2:
                test_norm[i, :] = test_norm[i, :] * 0.0
   
        self.__data = []
        for i in range(240):
            feature = test_norm[:, i*9:i*9+9]
            self.__data.append(feature)

        return self
    
    def add_bias(self):
        '''
        flatten feature and add bias
        '''
        for i in range(len(self.__data)): 
            pm25 = self.__data[i][9, 8] ** 3
            #self.__data[i] = np.insert(self.__data[i], len(self.__data[i]), pm25) 
            self.__data[i] = np.insert(self.__data[i], 0, 1)

    def __getitem__(self, index):
        return self.__data[index]
    
    def __len__(self):
        return len(self.__data) 
