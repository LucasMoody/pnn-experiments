import datasets.universal_dependencies_pos.UDPos as UDPos
import datasets.conll_ner.CoNLLNer as CoNLLNer
import models.POS.SennaPOS as POS
import models.NER.SennaNER as NER
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
    word2Idx = Embeddings.word2Idx
    # load params
    if learning_params is None:
        params = params_pos_ws_0
    else:
        params = learning_params

    n_in = 2 * params['window_size'] + 1

    # load dataset
    '''(ner_train_x, ner_train_case_x, ner_train_y, ner_train_y_cat), (ner_dev_x, ner_dev_case_x, ner_dev_y), (
       ner_test_x, ner_test_case_x, ner_test_y) = CoNLLNer.readDataset(params['window_size'], word2Idx, case2Idx)'''
    [ner_input_train, ner_train_y_cat], [ner_input_dev, ner_dev_y], [ner_input_test, ner_test_y], ner_dicts = CoNLLNer.readDataset(params['window_size'], word2Idx, case2Idx)
    [ner_train_x, ner_train_case_x] = ner_input_train
    [ner_dev_x, ner_dev_case_x] = ner_input_dev
    [ner_test_x, ner_test_case_x] = ner_input_test
    [word2Idx, caseLookup, ner_label2Idx, ner_idx2Label] = ner_dicts
    ner_n_out = ner_train_y_cat.shape[1]

    model_train_input_ner = [ner_train_x, ner_train_case_x]
    model_dev_input_ner = [ner_dev_x, ner_dev_case_x]
    model_test_input_ner = [ner_test_x, ner_test_case_x]

    n_in_x = ner_train_x.shape[1]
    n_in_casing = ner_train_case_x.shape[1]

    # ----- Build Model ----- #
    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing)
    model_ner = NER.buildNERModelGivenInput(input_layers, inputs, params, ner_n_out)

    print ner_train_x.shape[0], ' train samples'
    print ner_train_x.shape[1], ' train dimension'
    print ner_test_x.shape[0], ' test samples'

    # ----- Train Model ----- #
    iof1 = Measurer.create_compute_IOf1(ner_idx2Label)
    best_dev_scores, best_test_scores = Trainer.trainModel(model_ner, model_train_input_ner,
                                                                               ner_train_y_cat, params['number_of_epochs'],
                                                                               params['batch_size'],
                                                                               model_dev_input_ner,
                                                                               ner_dev_y, model_test_input_ner,
                                                                               ner_test_y, measurements=[iof1])
    return model_ner, best_dev_scores, best_test_scores

def getPOSModel(learning_params = None):
    if learning_params is None:
        params = params_pos_ws_0
    else:
        params = learning_params
    # Read in files

    n_in = 2 * params['window_size'] + 1

    [pos_input_train, pos_train_y_cat], [pos_input_dev, pos_dev_y], [pos_input_test, pos_test_y] = UDPos.readDataset(params['window_size'], word2Idx, case2Idx)
    pos_n_out = pos_train_y_cat.shape[1]

    [pos_train_x, pos_train_case_x] = pos_input_train
    [pos_dev_x, pos_dev_case_x] = pos_input_dev
    [pos_test_x, pos_test_case_x] = pos_input_test

    n_in_x = pos_train_x.shape[1]
    n_in_casing = pos_train_case_x.shape[1]

    # ----- Build Model ----- #
    #model_pos = POS.buildPosModel(n_in, embeddings, n_in_case, params, pos_n_out, metrics=metrics)
    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing)
    model_pos = POS.buildPosModelGivenInput(input_layers, inputs, params, pos_n_out)

    print pos_train_x.shape[0], ' train samples'
    print pos_train_x.shape[1], ' train dimension'
    print pos_test_x.shape[0], ' test samples'

    # ----- Train Model ----- #
    best_dev_scores, best_test_scores = Trainer.trainModel(model_pos, pos_input_train, pos_train_y_cat,
                                                             params['number_of_epochs'], params['batch_size'], pos_input_dev,
                                                             pos_dev_y, pos_input_test, pos_test_y, measurements=[Measurer.measureAccuracy])

    return model_pos, best_dev_scores, best_test_scores

def getPOSModelGivenInput(input_layers, inputs, learning_params = None, window_size = None):
    if learning_params is None:
        params = pos_default_params[window_size]
    else:
        params = learning_params

    # Read in files
    [pos_input_train, pos_train_y_cat], [pos_input_dev, pos_dev_y], [pos_input_test, pos_test_y] = UDPos.readDataset(
        params['window_size'], word2Idx, case2Idx)
    pos_n_out = pos_train_y_cat.shape[1]

    [pos_train_x, pos_train_case_x] = pos_input_train
    [pos_dev_x, pos_dev_case_x] = pos_input_dev
    [pos_test_x, pos_test_case_x] = pos_input_test

    # ----- Build Model ----- #
    model_pos = POS.buildPosModelGivenInput(input_layers, inputs, params, pos_n_out)

    print pos_train_x.shape[0], ' train samples'
    print pos_train_x.shape[1], ' train dimension'
    print pos_test_x.shape[0], ' test samples'

    # ----- Train Model ----- #
    best_dev_scores, best_test_scores = Trainer.trainModel(model_pos, pos_input_train, pos_train_y_cat,
                                                           params['number_of_epochs'], params['batch_size'], pos_input_dev,
                                                           pos_dev_y, pos_input_test, pos_test_y,
                                                           measurements=[Measurer.measureAccuracy])

    return model_pos, best_dev_scores, best_test_scores

def getNERModelGivenInput(input_layers, inputs, learning_params = None, window_size = None):
    if learning_params is None:
        params = ner_default_params[window_size]
    else:
        params = learning_params
    word2Idx = Embeddings.word2Idx
    # Read in files
    [ner_input_train, ner_train_y_cat], [ner_input_dev, ner_dev_y], [ner_input_test,
                                                                     ner_test_y], ner_dicts = CoNLLNer.readDataset(
        params['window_size'], word2Idx, case2Idx)
    [ner_train_x, ner_train_case_x] = ner_input_train
    [ner_dev_x, ner_dev_case_x] = ner_input_dev
    [ner_test_x, ner_test_case_x] = ner_input_test
    [word2Idx, caseLookup, ner_label2Idx, ner_idx2Label] = ner_dicts
    ner_n_out = ner_train_y_cat.shape[1]

    model_train_input_ner = [ner_train_x, ner_train_case_x]
    model_dev_input_ner = [ner_dev_x, ner_dev_case_x]
    model_test_input_ner = [ner_test_x, ner_test_case_x]

    n_in_x = ner_train_x.shape[1]
    n_in_casing = ner_train_case_x.shape[1]

    # ----- Build Model ----- #
    model_ner = NER.buildNERModelGivenInput(input_layers, inputs, params, ner_n_out)

    print ner_train_x.shape[0], ' train samples'
    print ner_train_x.shape[1], ' train dimension'
    print ner_test_x.shape[0], ' test samples'

    # ----- Train Model ----- #
    iof1 = Measurer.create_compute_IOf1(ner_idx2Label)
    best_dev_scores, best_test_scores = Trainer.trainModel(model_ner, ner_input_train, ner_train_y_cat,
                                                           params['number_of_epochs'], params['batch_size'], ner_input_dev,
                                                           ner_dev_y, ner_input_test, ner_test_y,
                                                           measurements=[iof1])

    return model_ner, best_dev_scores, best_test_scores