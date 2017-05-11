import sys, pickle
import string
import word2vec
import nltk
import nltk.data
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from sklearn.manifold import TSNE
import seaborn as sns

def train_word2vec(filepath='./data/hp_allphrase.txt', size=100, window=5, neg=5):
    
    # Turn training data into as better input for word2vec
    #word2vec.word2phrase('./data/hp_all.txt', './data/hp_allphrase.txt', verbose=True)
    
    # Train model
    word2vec.word2vec(filepath, './model/hp.bin', size=size, window=window, negative=neg, verbose=False)

    # Cluster vectors
    #word2vec.word2clusters('./data/hp_all.txt', './model/hp_clusters.txt', size, verbose=False)

def main():
       
    # Train model
    train_word2vec('./data/hp_allphrase.txt', size=110, window=5)
    
    # Load models
    model = word2vec.load('./model/hp.bin')
    maxent = nltk.data.load('./model/maxent_treebank_pos_tagger/PY3/english.pickle')
    #average = nltk.data.load('./model/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle')

    vocabs = []                 
    vecs = []                   
    plot_num = 1000
    for vocab in model.vocab:
        vocabs.append(vocab)
        vecs.append(model[vocab])
    vecs = np.array(vecs)[:plot_num]
    vocabs = vocabs[:plot_num]
    
    '''
    Dimensionality Reduction
    '''
    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(vecs)

    '''
    Plotting
    '''
    # filtering
    use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
    puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]
    
    plt.figure()
    texts = []
    for i, label in enumerate(vocabs):
        pos = maxent.tag([label])
        if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
                and all(c not in label for c in puncts)):
            x, y = reduced[i, :]
            texts.append(plt.text(x, y, label))
            plt.scatter(x, y)

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

    #plt.savefig('hp.png', dpi=600)
    plt.show()
   

if __name__ == "__main__":
    main()
