import pickle
import matplotlib.pyplot as plt
import matplotlib
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
    matplotlib.rcParams.update({'font.size': 25})
    plt.figure(figsize=(40,30))
    plt.barh(range(len(words)), freqs, align='center')
    plt.yticks(range(len(words)), words)
    plt.title('Tags count')
    plt.savefig('tag_counts.png', dpi=60)

if __name__ == "__main__":
    main()
