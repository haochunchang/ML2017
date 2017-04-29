import sys, pickle
import string
import word2vec
import nltk
import nltk.data
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from sklearn.manifold import TSNE

def plot_embedding(X, words, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    
    plt.figure(figsize=(25, 20))
    texts = []
    for i in range(X.shape[0]):
        plt.plot(X[i, 0], X[i, 1], 'ro')
        texts.append(plt.text(X[i, 0], X[i, 1], words[i]))  
    
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='b', lw=0.5), force_text=0.8)
    #plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

    plt.show()

def train_word2vec(filepath='./data/hp_allphrase.txt', size=100, window=5, neg=5):
    
    # Turn training data into as better input for word2vec
    #word2vec.word2phrase('./data/hp_all.txt', './data/hp_allphrase.txt', verbose=True)
    
    # Train model
    word2vec.word2vec(filepath, './model/hp.bin', size=size, window=window, negative=neg, verbose=False)

    # Cluster vectors
    #word2vec.word2clusters('./data/hp_all.txt', './model/hp_clusters.txt', size, verbose=False)

def preprocess(filepath, outfilepath):
    '''
    Preprocess by eliminating punctuations and stopwords.
    '''
    hp = ''
    # Load in text
    with open(filepath, 'r') as f:
        for line in f:
            hp += line.rstrip().lower()
    
    tokenizer = nltk.data.load('./model/punkt/PY3/english.pickle')
    tokens = tokenizer.tokenize(hp)
 
    # Remove puctuations
    transtab = {}
    for p in string.punctuation+'’—”“':
        transtab[ord(p)] = None
   
    no_punc = []
    for t in tokens:
        no_punc.append(t.translate(transtab))
    
    # Saved preprocessed file
    with open(outfilepath, 'w') as f:
        for line in no_punc:
            f.write(line+'\n')

def main():
    
    # Preprocess text
    preprocess('./data/hp_allphrase.txt', './data/hp_processed.txt')    
    
    # Train model
    train_word2vec('./data/hp_processed.txt', size=200, window=5)
    
    # Load models
    model = word2vec.load('./model/hp.bin')
    maxent = nltk.data.load('./model/maxent_treebank_pos_tagger/PY3/english.pickle')
    #average = nltk.data.load('./model/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle')

    # Stop word list
    stopwords = []
    with open('./model/stopwords/english', 'r') as f:
        for line in f:
            stopwords.append(line)

    tags = maxent.tag(model.vocab) 
    filtered = [word for word, tag in tags if tag in ['JJ','NNP','NN','NNS'] and len(word) > 1]
    filtered = [word for word in filtered if word not in stopwords]
    
    top_vectors = []
    for w in filtered:
        top_vectors.append(model[w])
    top_vectors = np.array(top_vectors).reshape(len(top_vectors), 200)

    # Project top vectors to 2-D space by TSNE
    embedding = TSNE(n_iter=1000)
    X = embedding.fit_transform(top_vectors[:500])
    plot_embedding(X, filtered, title='Visualization of Harry Potter Word2Vec')
    

if __name__ == "__main__":
    main()
