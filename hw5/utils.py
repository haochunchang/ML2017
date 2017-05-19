import os, pickle
import numpy as np
import keras
import word2vec
from keras.preprocessing import sequence, text
from keras.models import model_from_json

def load_embedding(embed_dim=100, pretrain='word2vec'):

    # Load Glove pre-trained model and preprocessing 
    if pretrain == 'glove':    
        embeddings_index = {}
 
        f = open(os.path.join('data', 'glove.6B.{}d.txt'.format(embed_dim)))
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

    # Load word2vec model only trained with train_data and test_data    
    elif pretrain == "word2vec":
        model = word2vec.load('./models/wordvec.bin')
   
        # Prepare embedding dictionary
        embeddings_index = {}
        for i in range(model.vocab.shape[0]):
            vocab, vector = model.vocab[i], model.vectors[i]
            word = vocab
            coefs = np.asarray(vector, dtype='float32')
            embeddings_index[word] = coefs

    # Load tokenizer
    with open('text_tokenizer.pkl', 'rb') as t:
        tokenizer = pickle.load(t)

    word_index = tokenizer.word_index
    # Look up dictionary of each words
    embedding_matrix = np.zeros((len(word_index) + 1, embed_dim))
    for word, i in word_index.items():
        if i < len(word_index) + 1:     
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def to_multi_categorical(tags,tags_list): 
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data

def preprocess(train_path=None, test_path=None, save_token=False):
    
    ### read training and testing data
    (Y_data,X_data,tag_list) = read_data(train_path,True)   
    y_train = to_multi_categorical(Y_data,tag_list) 
    
    if save_token:
        (_, X_test,_) = read_data(test_path,False)
        all_corpus = X_data + X_test
        print ('Find %d articles.' %(len(all_corpus)))
 
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(all_corpus)
    
        # Turn data to sequences and padding
        seqs = tokenizer.texts_to_sequences(X_data)    
        
        with open('text_tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)
        with open('tag_mapping.pkl', 'wb') as f:
            pickle.dump(tag_list, f)
    else:
        print('Use previous tokenizer.')
        with open('text_tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
    
        # Turn data to sequences and padding
        seqs = tokenizer.texts_to_sequences(X_data)    
           
    return seqs, y_train


def dump_history(store_path,logs):
    with open(os.path.join(store_path,'train_loss'),'a') as f:
        for loss in logs.tr_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'train_accuracy'),'a') as f:
        for acc in logs.tr_accs:
            f.write('{}\n'.format(acc))
    with open(os.path.join(store_path,'valid_loss'),'a') as f:
        for loss in logs.val_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'valid_accuracy'),'a') as f:
        for acc in logs.val_accs:
            f.write('{}\n'.format(acc))

if __name__ == "__main__":
    load_embedding(100, pretrain='glove')
