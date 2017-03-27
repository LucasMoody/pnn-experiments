from keras.layers import Input, Embedding, Flatten, Dense, merge, Dropout
from keras.models import Model
import theano
import numpy as np
from transfer import TransferUtils

#####################################
#
# Create the Keras Network for NER
#
#####################################

def buildNERModelGivenInput(input_layers, inputs, params, ner_n_out, metrics=[], useHiddenWeights=False, usePNN=False, additional_models=[]):


    ner_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='ner_hidden')
    ner_hidden = ner_hidden_layer(input_layers)

    ner_output_layer = Dense(output_dim=ner_n_out, activation='softmax', name='ner_output')
    ner_hidden_dropout = Dropout(params['dropout'])(ner_hidden)
    ner_output = ner_output_layer(ner_hidden_dropout)

    model = Model(input=inputs, output=[ner_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model

def buildNERModelWithPNN(input_layers, inputs, params, ner_n_out, metrics=[], additional_models=[]):
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

    ner_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='ner_hidden')
    ner_hidden = ner_hidden_layer(embeddings_hidden_merged)

    ner_hidden_merged = merge([ner_hidden] + transfer_model_output_layers, mode='concat')
    ner_hidden_dropout = Dropout(params['dropout'])(ner_hidden_merged)

    ner_output_layer = Dense(output_dim=ner_n_out, activation='softmax', name='ner_output')
    ner_output = ner_output_layer(ner_hidden_dropout)

    model = Model(input=inputs, output=[ner_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model

def buildNERModelWithAdapterPNN(input_layers, inputs, params, ner_n_out, metrics=[], additional_models=[]):
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

    adapter_layer = Dense(10, activation=params['activation'], name='ner_adapter')
    if(len(transfer_model_hidden_layers) > 1):
        adapter = adapter_layer(merge(transfer_model_hidden_layers, mode='concat'))
    else:
        adapter = adapter_layer(transfer_model_hidden_layers[0])

    embeddings_hidden_merged = merge([input_layers, adapter], mode='concat')

    ner_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='ner_hidden')
    ner_hidden = ner_hidden_layer(embeddings_hidden_merged)

    ner_hidden_merged = merge([ner_hidden] + transfer_model_output_layers, mode='concat')
    ner_hidden_dropout = Dropout(params['dropout'])(ner_hidden_merged)

    ner_output_layer = Dense(output_dim=ner_n_out, activation='softmax', name='ner_output')
    ner_output = ner_output_layer(ner_hidden_dropout)

    model = Model(input=inputs, output=[ner_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model

def buildNERModelWithDropoutPNN(input_layers, inputs, params, ner_n_out, metrics=[], additional_models=[]):
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

    pnn_dropout_hidden_layer = Dropout(0.3, name='ner_pnn_hidden_dropout')
    if (len(transfer_model_hidden_layers) > 1):
        pnn_dropout_hidden = pnn_dropout_hidden_layer(merge(transfer_model_hidden_layers, mode='concat'))
    else:
        pnn_dropout_hidden = pnn_dropout_hidden_layer(transfer_model_hidden_layers[0])

    embeddings_hidden_merged = merge([input_layers, pnn_dropout_hidden], mode='concat')

    ner_hidden_layer = Dense(params['hidden_dims'], activation=params['activation'], name='ner_hidden')
    ner_hidden = ner_hidden_layer(embeddings_hidden_merged)

    pnn_dropout_output_layer = Dropout(0.3, name='ner_pnn_output_dropout')
    if (len(transfer_model_output_layers) > 1):
        pnn_dropout_output = pnn_dropout_output_layer(merge(transfer_model_output_layers, mode='concat'))
    else:
        pnn_dropout_output = pnn_dropout_output_layer(transfer_model_output_layers[0])

    ner_hidden_merged = merge([ner_hidden, pnn_dropout_output], mode='concat')
    ner_hidden_dropout = Dropout(params['dropout'])(ner_hidden_merged)

    ner_output_layer = Dense(output_dim=ner_n_out, activation='softmax', name='ner_output')
    ner_output = ner_output_layer(ner_hidden_dropout)

    model = Model(input=inputs, output=[ner_output])

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=metrics)

    print model.summary()

    return model