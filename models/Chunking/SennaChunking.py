from keras.layers import Input, Embedding, Flatten, Dense, merge, Dropout
from keras.models import Model
import theano
import numpy as np
from transfer import TransferUtils

#####################################
#
# Create the Keras Network for chunking
#
#####################################

def buildChunkingModelGivenInput(input_layers, inputs, params, chunking_n_out, metrics=[], useHiddenWeights=False, additional_models=[]):

    if(useHiddenWeights):
        chunking_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='chunking_hidden',
                                      weights=TransferUtils.getHiddenLayerWeights(additional_models[0]))
    else:
        chunking_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='chunking_hidden')
    chunking_hidden = chunking_hidden_layer(input_layers)

    chunking_output_layer = Dense(output_dim=chunking_n_out, activation='softmax', name='chunking_output')
    chunking_hidden_dropout = Dropout(params['dropout'])(chunking_hidden)
    chunking_output = chunking_output_layer(chunking_hidden_dropout)

    model = Model(input=inputs, output=[chunking_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model

def buildChunkingModelWithPNN(input_layers, inputs, params, chunking_n_out, metrics=[], additional_models=[]):
    transfer_model_hidden_layers = []
    transfer_model_output_layers = []

    for model in additional_models:
        num_layers = len(model.layers)
        hidden = model.layers[num_layers - 3].output
        output = model.layers[num_layers - 1].output

        transfer_model_hidden_layers.append(hidden)
        transfer_model_output_layers.append(output)

        model.layers[num_layers - 3].trainable = False
        model.layers[num_layers - 1].trainable = False

    embeddings_hidden_merged = merge([input_layers] + transfer_model_hidden_layers, mode='concat')

    chunking_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='chunking_hidden')
    chunking_hidden = chunking_hidden_layer(embeddings_hidden_merged)

    chunking_hidden_merged = merge([chunking_hidden] + transfer_model_output_layers, mode='concat')
    chunking_hidden_dropout = Dropout(params['dropout'])(chunking_hidden_merged)

    chunking_output_layer = Dense(output_dim=chunking_n_out, activation='softmax', name='chunking_output')
    chunking_output = chunking_output_layer(chunking_hidden_dropout)

    model = Model(input=inputs, output=[chunking_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model

def buildChunkingModelWithSimplePNN(input_layers, inputs, params, chunking_n_out, metrics=[], additional_models=[]):
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

    chunking_hidden_merged = merge([chunking_hidden, pos_hidden, chunking_hidden], mode='concat')
    chunking_hidden_dropout = Dropout(params['dropout'])(chunking_hidden_merged)

    chunking_output_layer = Dense(output_dim=chunking_n_out, activation='softmax', name='chunking_output')
    chunking_output = chunking_output_layer(chunking_hidden_dropout)

    model = Model(input=inputs, output=[chunking_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model

def buildChunkingModelWithAdapterPNN(input_layers, inputs, params, chunking_n_out, metrics=[], additional_models=[]):
    transfer_model_hidden_layers = []
    transfer_model_output_layers = []

    for model in additional_models:
        num_layers = len(model.layers)
        hidden = model.layers[num_layers - 3].output
        output = model.layers[num_layers - 1].output

        transfer_model_hidden_layers.append(hidden)
        transfer_model_output_layers.append(output)

        model.layers[num_layers - 3].trainable = False
        model.layers[num_layers - 1].trainable = False

    adapter_layer = Dense(10, activation=params['activation'], name='chunking_adapter')
    if (len(transfer_model_hidden_layers) > 1):
        adapter = adapter_layer(merge(transfer_model_hidden_layers, mode='concat'))
    else:
        adapter = adapter_layer(transfer_model_hidden_layers[0])

    embeddings_hidden_merged = merge([input_layers, adapter], mode='concat')

    chunking_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='chunking_hidden')
    chunking_hidden = chunking_hidden_layer(embeddings_hidden_merged)

    chunking_hidden_merged = merge([chunking_hidden] + transfer_model_output_layers, mode='concat')
    chunking_hidden_dropout = Dropout(params['dropout'])(chunking_hidden_merged)

    chunking_output_layer = Dense(output_dim=chunking_n_out, activation='softmax', name='chunking_output')
    chunking_output = chunking_output_layer(chunking_hidden_dropout)

    model = Model(input=inputs, output=[chunking_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model


def buildChunkingModelWithDropoutPNN(input_layers, inputs, params, chunking_n_out, metrics=[], additional_models=[]):
    transfer_model_hidden_layers = []
    transfer_model_output_layers = []

    for model in additional_models:
        num_layers = len(model.layers)
        hidden = model.layers[num_layers - 3].output
        output = model.layers[num_layers - 1].output

        transfer_model_hidden_layers.append(hidden)
        transfer_model_output_layers.append(output)

        model.layers[num_layers - 3].trainable = False
        model.layers[num_layers - 1].trainable = False

    pnn_dropout_hidden_layer = Dropout(0.3, name='chunking_pnn_hidden_dropout')
    if (len(transfer_model_hidden_layers) > 1):
        pnn_dropout_hidden = pnn_dropout_hidden_layer(merge(transfer_model_hidden_layers, mode='concat'))
    else:
        pnn_dropout_hidden = pnn_dropout_hidden_layer(transfer_model_hidden_layers[0])

    embeddings_hidden_merged = merge([input_layers, pnn_dropout_hidden], mode='concat')

    chunking_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='chunking_hidden')
    chunking_hidden = chunking_hidden_layer(embeddings_hidden_merged)

    pnn_dropout_output_layer = Dropout(0.3, name='chunking_pnn_output_dropout')
    if (len(transfer_model_output_layers) > 1):
        pnn_dropout_output = pnn_dropout_output_layer(merge(transfer_model_output_layers, mode='concat'))
    else:
        pnn_dropout_output = pnn_dropout_output_layer(transfer_model_output_layers[0])

    chunking_hidden_merged = merge([chunking_hidden, pnn_dropout_output], mode='concat')
    chunking_hidden_dropout = Dropout(params['dropout'])(chunking_hidden_merged)

    chunking_output_layer = Dense(output_dim=chunking_n_out, activation='softmax', name='chunking_output')
    chunking_output = chunking_output_layer(chunking_hidden_dropout)

    model = Model(input=inputs, output=[chunking_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model