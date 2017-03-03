import numpy as np
import random

class Feature(object):
    '''
    List of features. Each entry is a matrix of a single day.
    '''
    def __init__(self, data):
        self.__data, self.__label = self._preprocess(data)
        
    def _preprocess(self, train):
        feature_lst = []
        label = []
        train.drop(['測站'], axis=1, inplace=True)
        train.replace('NR', 0, inplace=True)
        
        for month in range(1, 13):
            for day in range(1, 21):
                single_day = train.groupby(['日期']).get_group("2014/"+str(month)+"/"+str(day))
                single_day = single_day.drop(['日期', '測項'], axis=1)
                for hr in range(14):
                    feature_lst.append(np.array(single_day.iloc[:, hr:hr+9], dtype=np.float32))
                    label.append(int(single_day.iloc[9, hr+10]))

        # Normalization
        for i in range(len(feature_lst)):
            f = feature_lst[i]
            mu = f.mean(axis=1)
            std = f.std(axis=1)
            for j in range(len(mu)):
                if std[j] != 0:
                    feature_lst[i][j,] = (f[j,] - mu[j]) / std[j]

        return feature_lst, label
    
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

class TestFeature(Feature):
    '''
    Testing feature list with different preprocessing.
    '''
    def __init__(self, data):
        Feature.__init__(self, data)
        self.__data = self._preprocess(data)
        
    def _preprocess(self, test):
        feature_lst = []
        label = []
        test.replace('NR', 0, inplace=True)
           
        for i in range(240):
            single_day = test.groupby(0).get_group("id_"+str(i))
            single_day = single_day.drop(single_day.columns[:2], axis=1)
            feature_lst.append(np.array(single_day, dtype=np.float32))

        # Normalization
        for i in range(len(feature_lst)):
            f = feature_lst[i]
            mu = f.mean(axis=1)
            std = f.std(axis=1)
            for j in range(len(mu)):
                if std[j] != 0:
                    feature_lst[i][j,] = (f[j,] - mu[j]) / std[j]

        return feature_lst, label
    

