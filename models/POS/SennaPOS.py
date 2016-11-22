from keras.layers import Input, Embedding, Flatten, Dense, merge
from keras.models import Model
import theano
import numpy as np
from transfer import Extender

#####################################
#
# Create the Keras Network for POS
#
#####################################
def buildPosModel(n_in, embeddings, n_in_case, numHiddenUnitsPOS, pos_n_out, metrics=[], additional_models_for_input=[]):
    words_input = Input(shape=(n_in,), dtype='int32', name='words_input')
    wordEmbeddingLayer = Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in,  weights=[embeddings])
    words = wordEmbeddingLayer(words_input)
    words = Flatten(name='words_flatten')(words)

    caseMatrix = np.identity(n_in_case, dtype=theano.config.floatX)

    case_input = Input(shape=(n_in,), dtype='int32', name='case_input')
    caseEmbeddingLayer = Embedding(output_dim=caseMatrix.shape[1], input_dim=caseMatrix.shape[0], input_length=n_in, weights=[caseMatrix])
    casing = caseEmbeddingLayer(case_input)
    casing = Flatten(name='casing_flatten')(casing)

    input_layer = [words, casing]

    # for exp2
    additional_input_layers = Extender.buildInputLayerWithAdditionalModels(additional_models_for_input)

    input_layer.extend(additional_input_layers)

    words_casing_merged = merge(input_layer, mode='concat')
    pos_hidden_layer = Dense(numHiddenUnitsPOS, activation='tanh', name='pos_hidden')
    pos_hidden = pos_hidden_layer(words_casing_merged)

    pos_output_layer = Dense(output_dim=pos_n_out, activation='softmax', name='pos_output')
    pos_output = pos_output_layer(pos_hidden)

    input = [words_input, case_input]

    #for ex2
    input.extend(additional_input_layers)

    model = Model(input=input, output=[pos_output])

    #Don't update embeddings
    wordEmbeddingLayer.trainable_weights = []
    caseEmbeddingLayer.trainable_weights = []

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)

    print model.summary()

    return model