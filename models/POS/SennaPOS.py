from keras.layers import Input, Embedding, Flatten, Dense, merge, Dropout
from keras.models import Model
import theano
import numpy as np
from transfer import Extender

#####################################
#
# Create the Keras Network for POS
#
#####################################
def buildPosModel(n_in, embeddings, n_in_case, params, pos_n_out, metrics=[], additional_models_for_input=[], useModelInput=False, useHiddenWeights=False):
    words_input = Input(shape=(n_in,), dtype='int32', name='words_input')
    wordEmbeddingLayer = Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in,  weights=[embeddings], trainable=params['update_word_embeddings'])
    words = wordEmbeddingLayer(words_input)
    words = Flatten(name='words_flatten')(words)

    case_input = Input(shape=(n_in,), dtype='int32', name='case_input')
    caseEmbeddingLayer = Embedding(output_dim=n_in_case, input_dim=n_in_case, input_length=n_in,)
    casing = caseEmbeddingLayer(case_input)
    casing = Flatten(name='casing_flatten')(casing)

    input_layer = [words, casing]

    # for exp2
    if (useModelInput):
        additional_input_layers = Extender.buildInputLayerWithAdditionalModels(additional_models_for_input)
        input_layer.extend(additional_input_layers)

    words_casing_merged = merge(input_layer, mode='concat')
    # for exp3
    if (useHiddenWeights):
        pos_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='pos_hidden',
                                 weights=Extender.getHiddenLayerWeights(additional_models_for_input[0]))
    else:
        pos_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='pos_hidden')

    pos_hidden = pos_hidden_layer(words_casing_merged)

    pos_hidden_dropout_layer = Dropout(params['dropout'])
    pos_hidden_dropout = pos_hidden_dropout_layer(pos_hidden)

    pos_output_layer = Dense(output_dim=pos_n_out, activation='softmax', name='pos_output')
    pos_output = pos_output_layer(pos_hidden_dropout)

    input = [words_input, case_input]

    # for ex2
    if (useModelInput):
        input.extend(additional_input_layers)

    model = Model(input=input, output=[pos_output])

    #Don't update embeddings
    wordEmbeddingLayer.trainable_weights = []

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model

def buildPosModelGivenInput(input_layers, inputs, params, pos_n_out, useHiddenWeights=False, additional_models=[]):

    if(useHiddenWeights):
        pos_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='pos_hidden',
                                 weights=Extender.getHiddenLayerWeights(additional_models[0]))
    else:
        pos_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='pos_hidden')
    pos_hidden = pos_hidden_layer(input_layers)

    pos_hidden_dropout = Dropout(params['dropout'])(pos_hidden)

    pos_output_layer = Dense(output_dim=pos_n_out, activation='softmax', name='pos_output')
    pos_output = pos_output_layer(pos_hidden_dropout)

    model = Model(input=inputs, output=[pos_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'])

    print model.summary()

    return model