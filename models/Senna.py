from keras.layers import Input, Embedding, Flatten, Dense, merge, Dropout
from keras.models import Model
import theano
import numpy as np
from transfer import TransferUtils

def buildModelGivenInput(input_layers, inputs, params, n_out, metrics=[], useHiddenWeights=False, additional_models=[], name_prefix=''):

    input_dropout_layer = Dropout(params['dropout'])
    if(useHiddenWeights):
        hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name=name_prefix + 'hidden',
                             weights=TransferUtils.getHiddenLayerWeights(additional_models[0]))
    else:
        hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name=name_prefix + 'hidden')
    input_dropout = input_dropout_layer(input_layers)
    hidden = hidden_layer(input_dropout)

    output_layer = Dense(output_dim=n_out, activation='softmax', name=name_prefix + 'output')
    hidden_dropout = Dropout(params['dropout'])(hidden)
    output = output_layer(hidden_dropout)

    model = Model(input=inputs, output=[output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model

def buildModelWithFinetuning(input_layers, inputs, params, n_out, transfer_model, name_prefix=''):

    input_dropout_layer = Dropout(params['dropout'])
    hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name=name_prefix + 'hidden',
                         weights=getHiddenLayerWeights(transfer_model))
    input_dropout = input_dropout_layer(input_layers)
    hidden = hidden_layer(input_dropout)

    output_layer = Dense(output_dim=n_out, activation='softmax', name=name_prefix + 'output')
    hidden_dropout = Dropout(params['dropout'])(hidden)
    output = output_layer(hidden_dropout)

    model = Model(input=inputs, output=[output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'])

    print model.summary()

    return model

def buildModelWithPNN(input_layers, inputs, params, n_out, metrics=[], additional_models=[], name_prefix=''):
    # ----- GET TENSOR OF TRANSFER MODELS ----- #
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

    # MERGE INPUT WITH HIDDEN LAYERS OF TRANSFERRED MODELS
    embeddings_hidden_merged = merge([input_layers] + transfer_model_hidden_layers, mode='concat')

    # INPUT DROPOUT
    input_dropout_layer = Dropout(params['dropout'])
    input_dropout = input_dropout_layer(embeddings_hidden_merged)

    # HIDDEN LAYER
    hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name=name_prefix + 'hidden')
    hidden = hidden_layer(input_dropout)

    # MERGE HIDDEN WITH OUTPUT LAYERS OF TRANSFERRED MODELS
    hidden_merged = merge([hidden] + transfer_model_output_layers, mode='concat')
    hidden_dropout = Dropout(params['dropout'])(hidden_merged)

    # OUTPUT LAYER
    output_layer = Dense(output_dim=n_out, activation='softmax', name=name_prefix + 'output')
    output = output_layer(hidden_dropout)

    model = Model(input=inputs, output=[output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model

def buildModelWithSimplePNN(input_layers, inputs, params, n_out, metrics=[], additional_models=[], name_prefix=''):
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

    # INPUT DROPOUT
    input_dropout_layer = Dropout(params['dropout'])
    input_dropout = input_dropout_layer(input_layers)

    hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name=name_prefix + 'hidden')
    hidden = hidden_layer(input_dropout)

    # only use hidden layer
    hidden_merged = merge([hidden] + transfer_model_hidden_layers, mode='concat')
    hidden_dropout = Dropout(params['dropout'])(hidden_merged)

    output_layer = Dense(output_dim=n_out, activation='softmax', name=name_prefix + 'output')
    output = output_layer(hidden_dropout)

    model = Model(input=inputs, output=[output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model

def buildModelWithAdapterPNN(input_layers, inputs, params, n_out, metrics=[], additional_models=[], name_prefix=''):
    adapter_size = 10
    # ----- GET TENSOR OF TRANSFER MODELS ----- #
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

    # INPUT DROPOUT
    input_dropout_layer = Dropout(params['dropout'])
    input_dropout = input_dropout_layer(input_layers)

    # ADAPTER - HIDDEN #
    hidden_adapter_layer = Dense(adapter_size, activation=params['activation'], name=name_prefix + 'hidden_adapter')
    if (len(transfer_model_hidden_layers) > 1):
        hidden_adapter = hidden_adapter_layer(merge(transfer_model_hidden_layers, mode='concat'))
    else:
        hidden_adapter = hidden_adapter_layer(transfer_model_hidden_layers[0])

    # HIDDEN LAYER #
    embeddings_hidden_merged = merge([input_dropout, hidden_adapter], mode='concat')

    hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name=name_prefix + 'hidden')
    hidden = hidden_layer(embeddings_hidden_merged)

    # ADAPTER - OUTPUT
    output_adapter_layer = Dense(adapter_size, activation=params['activation'], name=name_prefix + 'output_adapter')
    if (len(transfer_model_output_layers) > 1):
        output_adapter = output_adapter_layer(merge(transfer_model_output_layers, mode='concat'))
    else:
        output_adapter = output_adapter_layer(transfer_model_output_layers[0])

    # OUTPUT LAYER
    hidden_merged = merge([hidden, output_adapter], mode='concat')
    hidden_dropout = Dropout(params['dropout'])(hidden_merged)

    output_layer = Dense(output_dim=n_out, activation='softmax', name=name_prefix + 'output')
    output = output_layer(hidden_dropout)

    model = Model(input=inputs, output=[output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model


def buildModelWithDropoutPNN(input_layers, inputs, params, n_out, metrics=[], additional_models=[], name_prefix=''):
    # ----- GET TENSOR OF TRANSFER MODELS ----- #
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

    # dropout layer for inputs (visible dropout)
    input_dropout_layer = Dropout(params['dropout'])
    input_dropout = input_dropout_layer(input_layers)

    # dropout layer for lateral connection to hidden layers of other NNs (PNN approach)
    pnn_dropout_hidden_layer = Dropout(params['dropout'], name=name_prefix + 'pnn_hidden_dropout')
    if (len(transfer_model_hidden_layers) > 1):
        pnn_dropout_hidden = pnn_dropout_hidden_layer(merge(transfer_model_hidden_layers, mode='concat'))
    else:
        pnn_dropout_hidden = pnn_dropout_hidden_layer(transfer_model_hidden_layers[0])

    # connect hidden layers from other PNNs with input layer
    embeddings_hidden_merged = merge([input_dropout, pnn_dropout_hidden], mode='concat')

    hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name=name_prefix + 'hidden')
    hidden = hidden_layer(embeddings_hidden_merged)

    # dropout layer for lateral connection to output layers of other NNs
    pnn_dropout_output_layer = Dropout(params['dropout'], name=name_prefix + 'pnn_output_dropout')
    if (len(transfer_model_output_layers) > 1):
        pnn_dropout_output = pnn_dropout_output_layer(merge(transfer_model_output_layers, mode='concat'))
    else:
        pnn_dropout_output = pnn_dropout_output_layer(transfer_model_output_layers[0])

    # connect connect output layers of other NNs with hidden layer of current model
    hidden_merged = merge([hidden, pnn_dropout_output], mode='concat')
    hidden_dropout = Dropout(params['dropout'])(hidden_merged)

    output_layer = Dense(output_dim=n_out, activation='softmax', name=name_prefix + 'output')
    output = output_layer(hidden_dropout)

    model = Model(input=inputs, output=[output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model

def buildModelWithOutput(input_layers, inputs, params, n_out, metrics=[], additional_models=[], name_prefix=''):
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

    embeddings_hidden_merged = merge([input_layers] + transfer_model_output_layers, mode='concat')

    input_dropout_layer = Dropout(params['dropout'])
    input_dropout = input_dropout_layer(embeddings_hidden_merged)

    hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name=name_prefix + 'hidden')
    hidden = hidden_layer(input_dropout)

    hidden_dropout = Dropout(params['dropout'])(hidden)

    output_layer = Dense(output_dim=n_out, activation='softmax', name=name_prefix + 'output')
    output = output_layer(hidden_dropout)

    model = Model(input=inputs, output=[output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model

def buildMultiTaskModelGivenInput(input_layers, inputs, params, model_info):
    models = []

    input_dropout_layer = Dropout(params['dropout'])
    hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='hidden')
    input_dropout = input_dropout_layer(input_layers)
    hidden = hidden_layer(input_dropout)

    hidden_dropout = Dropout(params['dropout'])(hidden)

    for name, n_out in model_info:
        second_hidden_layer = Dense(100, activation=params['activation'], name=name + 'second hidden')
        second_hidden = second_hidden_layer(hidden_dropout)

        second_hidden_dropout = Dropout(params['dropout'])(second_hidden)

        output_layer = Dense(output_dim=n_out, activation='softmax', name=name + 'output')
        output = output_layer(second_hidden_dropout)

        model = Model(input=inputs, output=[output])

        model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'])

        print model.summary()

        models.append(model)

    return models

def getHiddenLayerWeights(model):
    num_layers = len(model.layers)
    hidden_layer = model.layers[num_layers - 3]
    return [hidden_layer.W.get_value(), hidden_layer.b.get_value()]