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

def buildChunkingModelGivenInput(input_layers, inputs, params, ner_n_out, metrics=[], useHiddenWeights=False, additional_models=[]):

    if(useHiddenWeights):
        chunking_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='chunking_hidden',
                                 weights=Extender.getHiddenLayerWeights(additional_models[0]))
    else:
        chunking_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='chunking_hidden')
    chunking_hidden = chunking_hidden_layer(input_layers)

    chunking_output_layer = Dense(output_dim=ner_n_out, activation='softmax', name='chunking_output')
    chunking_hidden_dropout = Dropout(params['dropout'])(chunking_hidden)
    chunking_output = chunking_output_layer(chunking_hidden_dropout)

    model = Model(input=inputs, output=[chunking_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model

def buildChunkingModelWithPNN2(input_layers, inputs, params, ner_n_out, metrics=[], additional_models=[]):
    pos_model = additional_models[0]
    pos_num_layers = len(pos_model.layers)
    pos_hidden = pos_model.layers[pos_num_layers - 3].output
    pos_output = pos_model.layers[pos_num_layers - 1].output

    ner_model = additional_models[1]
    ner_num_layers = len(ner_model.layers)
    ner_hidden = ner_model.layers[ner_num_layers - 3].output
    ner_output = ner_model.layers[ner_num_layers - 1].output

    pos_model.layers[pos_num_layers - 3].trainable = False
    pos_model.layers[pos_num_layers - 1].trainable = False
    ner_model.layers[ner_num_layers - 3].trainable = False
    ner_model.layers[ner_num_layers - 1].trainable = False

    embeddings_hidden_merged = merge([input_layers, pos_hidden, ner_hidden], mode='concat')

    chunking_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='chunking_hidden')
    chunking_hidden = chunking_hidden_layer(embeddings_hidden_merged)

    chunking_hidden_merged = merge([chunking_hidden, pos_output, ner_output], mode='concat')
    chunking_hidden_dropout = Dropout(params['dropout'])(chunking_hidden_merged)

    chunking_output_layer = Dense(output_dim=ner_n_out, activation='softmax', name='chunking_output')
    chunking_output = chunking_output_layer(chunking_hidden_dropout)

    model = Model(input=inputs, output=[chunking_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model

def buildChunkingModelWithSimplePNN(input_layers, inputs, params, ner_n_out, metrics=[], additional_models=[]):
    pos_model = additional_models[0]
    pos_num_layers = len(pos_model.layers)
    pos_hidden = pos_model.layers[pos_num_layers - 3].output

    ner_model = additional_models[1]
    ner_num_layers = len(ner_model.layers)
    ner_hidden = ner_model.layers[ner_num_layers - 3].output

    pos_model.layers[pos_num_layers - 3].trainable = False
    pos_model.layers[pos_num_layers - 1].trainable = False
    ner_model.layers[ner_num_layers - 3].trainable = False
    ner_model.layers[ner_num_layers - 1].trainable = False

    chunking_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='chunking_hidden')
    chunking_hidden = chunking_hidden_layer(input_layers)

    chunking_hidden_merged = merge([chunking_hidden, pos_hidden, ner_hidden], mode='concat')
    chunking_hidden_dropout = Dropout(params['dropout'])(chunking_hidden_merged)

    chunking_output_layer = Dense(output_dim=ner_n_out, activation='softmax', name='chunking_output')
    chunking_output = chunking_output_layer(chunking_hidden_dropout)

    model = Model(input=inputs, output=[chunking_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model
