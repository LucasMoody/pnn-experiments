import numpy as np
import theano

vocabPath = 'embeddings/dependency_based_word_embeddings/vocabs/levy_dependency_based.words.vocab'

def readEmbeddings():
    print "Read in the vocab"
    word2Idx = {} #Maps a word to the index in the embeddings matrix
    embeddings = [] #Embeddings matrix
    with open(vocabPath, 'r') as fIn:
        idx = 0
        for line in fIn:
            split = line.strip().split(' ')
            embeddings.append(np.array([float(num) for num in split[1:]]))
            word2Idx[split[0]] = idx
            idx += 1
    embeddings = np.asarray(embeddings, dtype=theano.config.floatX)
    return embeddings, word2Idx

embeddings, word2Idx = readEmbeddings()