import pickle
import matplotlib.pyplot as plt
from keras.preprocessing import text
import utils
import numpy as np
from collections import Counter
import operator

def main(): 
    (Y_data,X_data,tag_list) = utils.read_data('./data/train_data.csv',True)   
    Y_data = [item for sublist in Y_data for item in sublist]

    counts = Counter(Y_data)   
    d = counts
    sorted_d = sorted(d.items(), key=operator.itemgetter(1))
    print(sorted_d)
    words = []
    freqs = []
    for d in sorted_d:
        word, freq = d
        words.append(word)
        freqs.append(freq)

    plt.figure(figsize=(20,10))
    plt.barh(range(len(words)), freqs, align='center')
    plt.yticks(range(len(words)), words)
    plt.title('Tags count')
    plt.show()

if __name__ == "__main__":
    main()
