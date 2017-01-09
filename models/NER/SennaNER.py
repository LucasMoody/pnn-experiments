from keras.layers import Input, Embedding, Flatten, Dense, merge, Dropout
from keras.models import Model
import theano
import numpy as np
from transfer import Extender

#####################################
#
# Create the Keras Network for NER
#
#####################################
def buildNERModel(n_in, embeddings, n_in_case, params, ner_n_out, metrics=[], additional_models_for_input=[], useModelInput=False, useHiddenWeights=False):
    words_input = Input(shape=(n_in,), dtype='int32', name='words_input')
    wordEmbeddingLayer = Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in,  weights=[embeddings], trainable=False)
    words = wordEmbeddingLayer(words_input)
    words = Flatten(name='words_flatten')(words)

    caseMatrix = np.identity(n_in_case, dtype=theano.config.floatX)

    case_input = Input(shape=(n_in,), dtype='int32', name='case_input')
    caseEmbeddingLayer = Embedding(output_dim=caseMatrix.shape[1], input_dim=caseMatrix.shape[0], input_length=n_in, weights=[caseMatrix], trainable=False)
    casing = caseEmbeddingLayer(case_input)
    casing = Flatten(name='casing_flatten')(casing)

    input_layer = [words, casing]

    # for exp2
    if(useModelInput):
        additional_input_layers = Extender.buildInputLayerWithAdditionalModels(additional_models_for_input)
        input_layer.extend(additional_input_layers)

    words_casing_merged = merge(input_layer, mode='concat')
    # for exp3
    if(useHiddenWeights):
        ner_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='ner_hidden',
                                 weights=Extender.getHiddenLayerWeights(additional_models_for_input[0]))
    else:
        ner_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='ner_hidden')
    ner_hidden = ner_hidden_layer(words_casing_merged)

    ner_output_layer = Dense(output_dim=ner_n_out, activation='softmax', name='ner_output')
    ner_output = ner_output_layer(ner_hidden)

    input = [words_input, case_input]

    #for ex2
    if(useModelInput):
        input.extend(additional_input_layers)

    model = Model(input=input, output=[ner_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model


def buildNERModelGivenInput(input_layers, inputs, params, ner_n_out, metrics=[], useHiddenWeights=False, usePNN=False, additional_models=[]):

    if(useHiddenWeights):
        ner_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='ner_hidden',
                                 weights=Extender.getHiddenLayerWeights(additional_models[0]))
    else:
        ner_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='ner_hidden')
    ner_hidden = ner_hidden_layer(input_layers)

    ner_output_layer = Dense(output_dim=ner_n_out, activation='softmax', name='ner_output')
    if(usePNN):
        pos_model = additional_models[0]
        num_layers = len(pos_model.layers)
        pos_hidden = pos_model.get_layer(name='pos_hidden').output
        pos_hidden.trainable_weights = []
        ner_hidden_merged_layer = merge([ner_hidden, pos_hidden], mode='concat')
        ner_hidden_dropout = Dropout(params['dropout'])(ner_hidden_merged_layer)
    else:
        ner_hidden_dropout = Dropout(params['dropout'])(ner_hidden)
    ner_output = ner_output_layer(ner_hidden_dropout)

    model = Model(input=inputs, output=[ner_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model

def buildNERModelWithPNN2(input_layers, inputs, params, ner_n_out, metrics=[], additional_models=[]):
    pos_model = additional_models[0]
    num_layers = len(pos_model.layers)
    pos_hidden = pos_model.layers[num_layers - 3].output
    pos_output = pos_model.layers[num_layers - 1].output

    embeddings_hidden_merged = merge([input_layers, pos_hidden], mode='concat')

    ner_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='ner_hidden')
    ner_hidden = ner_hidden_layer(embeddings_hidden_merged)

    ner_hidden_merged = merge([ner_hidden, pos_output], mode='concat')
    ner_hidden_dropout = Dropout(params['dropout'])(ner_hidden_merged)

    ner_output_layer = Dense(output_dim=ner_n_out, activation='softmax', name='ner_output')
    ner_output = ner_output_layer(ner_hidden_dropout)

    pos_hidden.trainable_weights = []
    pos_output.trainable_weights = []

    model = Model(input=inputs, output=[ner_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model