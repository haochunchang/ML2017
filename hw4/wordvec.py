import sys
import word2vec

def main():
    '''
    # Turn training data into as better input for word2vec
    word2vec.word2phrase('./data/hp_all.txt', './data/hp_allphrase.txt', verbose=True)
    '''
    # Train model
    word2vec.word2vec('./data/hp_allphrase.txt', './model/hp.bin', size=100, verbose=False)

    # Cluster vectors
    word2vec.word2clusters('./data/hp_all.txt', './model/hp_clusters.txt', 100, verbose=False)
    
    model = word2vec.load('./model/hp.bin')
    print(model.vocab[:500])

if __name__ == "__main__":
    main()
