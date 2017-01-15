from datasets.wsj_pos import WSJPos
import datasets.conll_ner.CoNLLNer as CoNLLNer
import datasets.conll_chunking.CoNLLChunking as CoNLLChunking
import models.POS.SennaPOS as POS
import models.NER.SennaNER as NER
import models.Chunking.SennaChunking as Chunking
from models import Trainer, InputBuilder
import config

from embeddings.dependency_based_word_embeddings import DependencyBasedWordEmbeddings as Embeddings
from measurements import Measurer

# settings

params_quick = {
    'update_word_embeddings': True,
    'window_size': 0,
    'batch_size': 128,
    'hidden_dims': 180,
    'activation': 'relu',
    'dropout': 0.25,
    'optimizer': 'nadam',
    'number_of_epochs': 1
}

params_pos_ws_0 = {
    'update_word_embeddings': True,
    'window_size': 0,
    'batch_size': 128,
    'hidden_dims': 180,
    'activation': 'relu',
    'dropout': 0.25,
    'optimizer': 'nadam',
    'number_of_epochs': 11
}

params_pos_ws_1 = {
    'update_word_embeddings': True,
    'window_size': 1,
    'batch_size': 128,
    'hidden_dims': 300,
    'activation': 'relu',
    'dropout': 0.25,
    'optimizer': 'adamax',
    'number_of_epochs': 13
}

params_pos_ws_2 = {
    'update_word_embeddings': True,
    'window_size': 2,
    'batch_size': 32,
    'hidden_dims': 230,
    'activation': 'sigmoid',
    'dropout': 0.45,
    'optimizer': 'nadam',
    'number_of_epochs': 11
}

params_pos_ws_3 = {
    'update_word_embeddings': False,
    'window_size': 3,
    'batch_size': 64,
    'hidden_dims': 280,
    'activation': 'relu',
    'dropout': 0.6,
    'optimizer': 'adamax',
    'number_of_epochs': 19
}

params_pos_ws_4 = {
    'update_word_embeddings': False,
    'window_size': 4,
    'batch_size': 128,
    'hidden_dims': 255,
    'activation': 'sigmoid',
    'dropout': 0.65,
    'optimizer': 'nadam',
    'number_of_epochs': 19
}

params_ner_ws_0 = {
    'update_word_embeddings': True,
    'window_size': 0,
    'batch_size': 64,
    'hidden_dims': 185,
    'activation': 'relu',
    'dropout': 0.1,
    'optimizer': 'adam',
    'number_of_epochs': 14
}

params_ner_ws_1 = {
    'update_word_embeddings': False,
    'window_size': 1,
    'batch_size': 128,
    'hidden_dims': 235,
    'activation': 'relu',
    'dropout': 0.5,
    'optimizer': 'adam',
    'number_of_epochs': 16
}

params_ner_ws_2 = {
    'update_word_embeddings': True,
    'window_size': 2,
    'batch_size': 32,
    'hidden_dims': 270,
    'activation': 'sigmoid',
    'dropout': 0.4,
    'optimizer': 'adam',
    'number_of_epochs': 19
}

params_ner_ws_3 = {
    'update_word_embeddings': True,
    'window_size': 3,
    'batch_size': 32,
    'hidden_dims': 175,
    'activation': 'sigmoid',
    'dropout': 0.45,
    'optimizer': 'adam',
    'number_of_epochs': 19
}

params_ner_ws_4 = {
    'update_word_embeddings': False,
    'window_size': 4,
    'batch_size': 32,
    'hidden_dims': 190,
    'activation': 'sigmoid',
    'dropout': 0.5,
    'optimizer': 'nadam',
    'number_of_epochs': 9
}

pos_default_params = {
    0: params_pos_ws_0,
    1: params_pos_ws_1,
    2: params_pos_ws_2,
    3: params_pos_ws_3,
    4: params_pos_ws_4
}

ner_default_params = {
    0: params_ner_ws_0,
    1: params_ner_ws_1,
    2: params_ner_ws_2,
    3: params_ner_ws_3,
    4: params_ner_ws_4
}

metrics = []

# ----- metric results -----#
metric_results = []

#Casing matrix
case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING':7}
n_in_case = len(case2Idx)

word2Idx = Embeddings.word2Idx
embeddings = Embeddings.embeddings

def getNERModel(learning_params = None):
    # load params
    if learning_params is None:
        params = params_pos_ws_0
    else:
        params = learning_params

    # load dataset
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = CoNLLNer.readDataset(params['window_size'], word2Idx, case2Idx)
    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, caseLookup, label2Idx, idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    # ----- Build Model ----- #
    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing)
    model = NER.buildNERModelGivenInput(input_layers, inputs, params, n_out)

    print train_x.shape[0], ' train samples'
    print train_x.shape[1], ' train dimension'
    print test_x.shape[0], ' test samples'

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModel(model, input_train,
                                                                               train_y_cat, params['number_of_epochs'],
                                                                               params['batch_size'],
                                                                               input_dev,
                                                                               dev_y, input_test,
                                                                               test_y, measurements=[biof1])
    return train_scores, dev_scores, test_scores

def getPOSModel(learning_params = None):
    if learning_params is None:
        params = params_pos_ws_0
    else:
        params = learning_params

    # Read in files
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y] = WSJPos.readDataset(params['window_size'], word2Idx, case2Idx)
    n_out = train_y_cat.shape[1]

    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    # ----- Build Model ----- #
    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing)
    model = POS.buildPosModelGivenInput(input_layers, inputs, params, n_out)

    print train_x.shape[0], ' train samples'
    print train_x.shape[1], ' train dimension'
    print test_x.shape[0], ' test samples'

    # ----- Train Model ----- #
    train_scores, best_dev_scores, best_test_scores = Trainer.trainModel(model, input_train, train_y_cat,
                                                             params['number_of_epochs'], params['batch_size'], input_dev,
                                                             dev_y, input_test, test_y, measurements=[Measurer.measureAccuracy])

    return train_scores, best_dev_scores, best_test_scores

def getChunkingModel(learning_params = None):
    if learning_params is None:
        params = params_pos_ws_0
    else:
        params = learning_params

    word2Idx = Embeddings.word2Idx
    # ----- NER ----- #

    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts= CoNLLChunking.readDataset(params['window_size'], word2Idx, case2Idx)
    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [word2Idx, _, label2Idx, idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    # ----- Build Model ----- #
    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing)
    model = Chunking.buildChunkingModelGivenInput(input_layers, inputs, params, n_out)

    print train_x.shape[0], ' train samples'
    print train_x.shape[1], ' train dimension'
    print test_x.shape[0], ' test samples'


    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModel(model, input_train,
                                                           train_y_cat, params['number_of_epochs'],
                                                           params['batch_size'],
                                                           input_dev,
                                                           dev_y, input_test,
                                                           test_y, measurements=[biof1])


    return train_scores, dev_scores, test_scores

def getPOSModelGivenInput(input_layers, inputs, learning_params = None, window_size = None):
    if learning_params is None:
        params = pos_default_params[window_size]
        params['number_of_epochs'] = 1
    else:
        params = learning_params

    print params
    # Read in files
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y] = WSJPos.readDataset(
        params['window_size'], word2Idx, case2Idx)
    n_out = train_y_cat.shape[1]

    [train_x, pos_train_case_x] = input_train
    [dev_x, pos_dev_case_x] = input_dev
    [test_x, pos_test_case_x] = input_test

    # ----- Build Model ----- #
    model = POS.buildPosModelGivenInput(input_layers, inputs, params, n_out)

    print train_x.shape[0], ' train samples'
    print train_x.shape[1], ' train dimension'
    print test_x.shape[0], ' test samples'

    # ----- Train Model ----- #
    train_scores, dev_scores, test_scores = Trainer.trainModel(model, input_train, train_y_cat,
                                                           params['number_of_epochs'], params['batch_size'], input_dev,
                                                           dev_y, input_test, test_y,
                                                           measurements=[Measurer.measureAccuracy])

    return model, train_scores, dev_scores, test_scores

def getNERModelGivenInput(input_layers, inputs, learning_params = None, window_size = None):
    if learning_params is None:
        params = ner_default_params[window_size]
        #params['number_of_epochs'] = 1
    else:
        params = learning_params
    print params
    # Read in files
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = CoNLLNer.readDataset(
        params['window_size'], word2Idx, case2Idx)

    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, caseLookup, label2Idx, idx2Label] = dicts
    ner_n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    # ----- Build Model ----- #
    model = NER.buildNERModelGivenInput(input_layers, inputs, params, ner_n_out)

    print train_x.shape[0], ' train samples'
    print train_x.shape[1], ' train dimension'
    print test_x.shape[0], ' test samples'

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(idx2Label)
    train_scores, best_dev_scores, best_test_scores = Trainer.trainModel(model, input_train, train_y_cat,
                                                           params['number_of_epochs'], params['batch_size'], input_dev,
                                                           dev_y, input_test, test_y,
                                                           measurements=[biof1])

    return model, train_scores, best_dev_scores, best_test_scores

def getChunkingModelGivenInput(input_layers, inputs, learning_params = None, window_size = None):
    if learning_params is None:
        params = ner_default_params[window_size]
        #params['number_of_epochs'] = 1
    else:
        params = learning_params
    print params
    # ----- Chunking ----- #

    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts= CoNLLChunking.readDataset(params['window_size'], word2Idx, case2Idx)
    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, _, label2Idx, idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    # ----- Build Model ----- #
    model = Chunking.buildChunkingModelGivenInput(input_layers, inputs, params, n_out)

    print train_x.shape[0], ' train samples'
    print train_x.shape[1], ' train dimension'
    print test_x.shape[0], ' test samples'


    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModel(model, input_train, train_y_cat,
                                                           params['number_of_epochs'], params['batch_size'], input_dev,
                                                           dev_y, input_test, test_y,
                                                           measurements=[biof1])
    return model, train_scores, dev_scores, test_scores
