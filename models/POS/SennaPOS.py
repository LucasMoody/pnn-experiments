from keras.layers import Input, Embedding, Flatten, Dense, merge, Dropout
from keras.models import Model
import theano
import numpy as np
from transfer import TransferUtils

#####################################
#
# Create the Keras Network for POS
#
#####################################

def buildPosModelGivenInput(input_layers, inputs, params, pos_n_out, useHiddenWeights=False, additional_models=[]):

    if(useHiddenWeights):
        pos_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='pos_hidden',
                                 weights=TransferUtils.getHiddenLayerWeights(additional_models[0]))
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

def buildPOSModelWithPNN(input_layers, inputs, params, pos_n_out, metrics=[], additional_models=[], name_prefix=''):
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

    pos_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name=name_prefix + 'pos_hidden')
    pos_hidden = pos_hidden_layer(embeddings_hidden_merged)

    pos_hidden_merged = merge([pos_hidden] + transfer_model_output_layers, mode='concat')
    pos_hidden_dropout = Dropout(params['dropout'])(pos_hidden_merged)

    pos_output_layer = Dense(output_dim=pos_n_out, activation='softmax', name=name_prefix + 'pos_output')
    pos_output = pos_output_layer(pos_hidden_dropout)

    model = Model(input=inputs, output=[pos_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model

def buildPOSModelWithAdapterPNN(input_layers, inputs, params, pos_n_out, metrics=[], additional_models=[], name_prefix=''):
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

    pnn_dropout_hidden_layer = Dropout(0.3, name='pos_pnn_hidden_dropout')
    if (len(transfer_model_hidden_layers) > 1):
        pnn_dropout_hidden = pnn_dropout_hidden_layer(merge(transfer_model_hidden_layers, mode='concat'))
    else:
        pnn_dropout_hidden = pnn_dropout_hidden_layer(transfer_model_hidden_layers[0])

    embeddings_hidden_merged = merge([input_layers, pnn_dropout_hidden], mode='concat')

    pos_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name=name_prefix + 'pos_hidden')
    pos_hidden = pos_hidden_layer(embeddings_hidden_merged)

    pnn_dropout_output_layer = Dropout(0.3, name='pos_pnn_output_dropout')
    if (len(transfer_model_output_layers) > 1):
        pnn_dropout_output = pnn_dropout_output_layer(merge(transfer_model_output_layers, mode='concat'))
    else:
        pnn_dropout_output = pnn_dropout_output_layer(transfer_model_output_layers[0])

    pos_hidden_merged = merge([pos_hidden, pnn_dropout_output], mode='concat')
    pos_hidden_dropout = Dropout(params['dropout'])(pos_hidden_merged)

    pos_output_layer = Dense(output_dim=pos_n_out, activation='softmax', name=name_prefix + 'pos_output')
    pos_output = pos_output_layer(pos_hidden_dropout)

    model = Model(input=inputs, output=[pos_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model

def buildPOSModelWithDropoutPNN(input_layers, inputs, params, pos_n_out, metrics=[], additional_models=[], name_prefix=''):
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

    pnn_dropout_hidden_layer = Dropout(0.3, name='pos_pnn_hidden_dropout')
    if (len(transfer_model_hidden_layers) > 1):
        pnn_dropout_hidden = pnn_dropout_hidden_layer(merge(transfer_model_hidden_layers, mode='concat'))
    else:
        pnn_dropout_hidden = pnn_dropout_hidden_layer(transfer_model_hidden_layers[0])

    embeddings_hidden_merged = merge([input_layers, pnn_dropout_hidden], mode='concat')

    pos_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name=name_prefix + 'pos_hidden')
    pos_hidden = pos_hidden_layer(embeddings_hidden_merged)

    pnn_dropout_output_layer = Dropout(0.3, name='pos_pnn_output_dropout')
    if (len(transfer_model_output_layers) > 1):
        pnn_dropout_output = pnn_dropout_output_layer(merge(transfer_model_output_layers, mode='concat'))
    else:
        pnn_dropout_output = pnn_dropout_output_layer(transfer_model_output_layers[0])

    pos_hidden_merged = merge([pos_hidden, pnn_dropout_output], mode='concat')
    pos_hidden_dropout = Dropout(params['dropout'])(pos_hidden_merged)

    pos_output_layer = Dense(output_dim=pos_n_out, activation='softmax', name=name_prefix + 'pos_output')
    pos_output = pos_output_layer(pos_hidden_dropout)

    model = Model(input=inputs, output=[pos_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model