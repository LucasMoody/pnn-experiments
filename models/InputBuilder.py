from keras.layers import Input, Embedding, Flatten, Dense, merge
from keras.models import Model
import theano
import numpy as np

def buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing):
    words_input = Input(shape=(n_in_x,), dtype='int32', name='words_input')
    wordEmbeddingLayer = Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in_x,
                                   weights=[embeddings], trainable=False)
    words = wordEmbeddingLayer(words_input)
    words = Flatten(name='words_flatten')(words)

    case_input = Input(shape=(n_in_x,), dtype='int32', name='case_input')
    caseEmbeddingLayer = Embedding(output_dim=len(case2Idx), input_dim=len(case2Idx), input_length=n_in_casing,
                                   trainable=True)
    casing = caseEmbeddingLayer(case_input)
    casing = Flatten(name='casing_flatten')(casing)

    input_layers = [words, casing]
    inputs = [words_input, case_input]

    input_layers_merged = merge(input_layers, mode='concat')

    return input_layers_merged, inputs