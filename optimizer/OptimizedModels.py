import datasets.universal_dependencies_pos.UDPos as UDPos
import datasets.conll_ner.CoNLLNer as CoNLLNer
import models.POS.SennaPOS as POS
import models.NER.SennaNER as NER
from models import Trainer, InputBuilder
import config

from embeddings.dependency_based_word_embeddings import DependencyBasedWordEmbeddings as Embeddings
from measurements import Measurer

# settings
default_params = {
    'update_word_embeddings': False,
    'window_size': 3,
    'batch_size': 128,
    'hidden_dims': 100,
    'activation': 'tanh',
    'dropout': 0.3,
    'optimizer': 'adam'
}


'''windowSize = 3 # n to the left, n to the right
n_in = 2 * windowSize + 1
numHiddenUnitsPOS = 100
numHiddenUnitsNER = 100
n_minibatches = 1000'''
number_of_epochs = config.number_of_epochs
metrics = []

# ----- metric results -----#
metric_results = []

#Casing matrix
case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING':7}
n_in_case = len(case2Idx)

embeddings = Embeddings.embeddings
word2Idx = Embeddings.word2Idx

def getNERModel(learning_params = None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    n_in = 2 * params['window_size'] + 1

    #   ----- NER ----- #
    (ner_train_x, ner_train_case_x, ner_train_y, ner_train_y_cat), (ner_dev_x, ner_dev_case_x, ner_dev_y), (
        ner_test_x, ner_test_case_x, ner_test_y) = CoNLLNer.readDataset(params['window_size'], word2Idx, case2Idx)
    ner_n_out = ner_train_y_cat.shape[1]

    model_train_input_ner = [ner_train_x, ner_train_case_x]
    model_dev_input_ner = [ner_dev_x, ner_dev_case_x]
    model_test_input_ner = [ner_test_x, ner_test_case_x]


    # ----- Build Model ----- #
    model_ner = NER.buildNERModel(n_in, embeddings, n_in_case, params, ner_n_out, metrics=metrics)

    print ner_train_x.shape[0], ' train samples'
    print ner_train_x.shape[1], ' train dimension'
    print ner_test_x.shape[0], ' test samples'

    # ----- Train Model ----- #

    best_dev_scores, best_test_scores = Trainer.trainModel(model_ner, model_train_input_ner,
                                                                               ner_train_y_cat, number_of_epochs,
                                                                               params['batch_size'],
                                                                               model_dev_input_ner,
                                                                               ner_dev_y, model_test_input_ner,
                                                                               ner_test_y, measurements=[Measurer.measureAccuracy])
    return model_ner, best_dev_scores, best_test_scores

def getPOSModel(learning_params = None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params
    # Read in files

    n_in = 2 * params['window_size'] + 1

    (pos_train_x, pos_train_case_x, pos_train_y, pos_train_y_cat), (pos_dev_x, pos_dev_case_x, pos_dev_y), (
        pos_test_x, pos_test_case_x, pos_test_y) = UDPos.readDataset(params['window_size'], word2Idx, case2Idx)
    pos_n_out = pos_train_y_cat.shape[1]

    model_train_input_pos = [pos_train_x, pos_train_case_x]
    model_dev_input_pos = [pos_dev_x, pos_dev_case_x]
    model_test_input_pos = [pos_test_x, pos_test_case_x]

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
    best_dev_scores, best_test_scores = Trainer.trainModel(model_pos, model_train_input_pos, pos_train_y_cat,
                                                             number_of_epochs, params['batch_size'], model_dev_input_pos,
                                                             pos_dev_y, model_test_input_pos, pos_test_y, measurements=[Measurer.measureAccuracy])

    return model_pos, best_dev_scores, best_test_scores

def getPOSModelGivenInput(input_layers, inputs, learning_params = None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    # Read in files
    (pos_train_x, pos_train_case_x, pos_train_y, pos_train_y_cat), (pos_dev_x, pos_dev_case_x, pos_dev_y), (
        pos_test_x, pos_test_case_x, pos_test_y) = UDPos.readDataset(params['window_size'], word2Idx, case2Idx)
    pos_n_out = pos_train_y_cat.shape[1]

    model_train_input_pos = [pos_train_x, pos_train_case_x]
    model_dev_input_pos = [pos_dev_x, pos_dev_case_x]
    model_test_input_pos = [pos_test_x, pos_test_case_x]

    # ----- Build Model ----- #
    model_pos = POS.buildPosModelGivenInput(input_layers, inputs, params, pos_n_out)

    print pos_train_x.shape[0], ' train samples'
    print pos_train_x.shape[1], ' train dimension'
    print pos_test_x.shape[0], ' test samples'

    # ----- Train Model ----- #
    best_dev_scores, best_test_scores = Trainer.trainModel(model_pos, model_train_input_pos, pos_train_y_cat,
                                                           number_of_epochs, params['batch_size'], model_dev_input_pos,
                                                           pos_dev_y, model_test_input_pos, pos_test_y,
                                                           measurements=[Measurer.measureAccuracy])

    return model_pos, best_dev_scores, best_test_scores