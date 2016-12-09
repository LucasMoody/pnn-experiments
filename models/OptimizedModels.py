import datasets.universal_dependencies_pos.UDPos as UDPos
import datasets.conll_ner.CoNLLNer as CoNLLNer
import embeddings.dependency_based_word_embeddings.DependencyBasedWordEmbeddings as Embeddings
from transfer import Extender
import numpy as np
import models.POS.SennaPOS as POS
import models.NER.SennaNER as NER
import models.Trainer as Trainer
from keras.layers import Input, Dense
from keras.models import Model

from keras.utils import np_utils
from measurements import Measurer
import plots.LearningCurve as LearningCurve

# settings
windowSize = 3 # n to the left, n to the right
n_in = 2 * windowSize + 1
numHiddenUnitsPOS = 100
numHiddenUnitsNER = 100
n_minibatches = 1000
number_of_epochs = 1
metrics = []

# ----- metric results -----#
metric_results = []

#Casing matrix
caseLookup = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING':7}
n_in_case = len(caseLookup)

def getNERModel(embeddings, word2Idx):
    #   ----- NER ----- #
    (ner_train_x, ner_train_case_x, ner_train_y, ner_train_y_cat), (ner_dev_x, ner_dev_case_x, ner_dev_y), (
        ner_test_x, ner_test_case_x, ner_test_y) = CoNLLNer.readDataset(windowSize, word2Idx, caseLookup)
    ner_n_out = ner_train_y_cat.shape[1]

    model_train_input_ner = [ner_train_x, ner_train_case_x]
    model_dev_input_ner = [ner_dev_x, ner_dev_case_x]
    model_test_input_ner = [ner_test_x, ner_test_case_x]


    # ----- Build Model ----- #
    model_ner = NER.buildNERModel(n_in, embeddings, n_in_case, numHiddenUnitsNER, ner_n_out, metrics=metrics)

    print ner_train_x.shape[0], ' train samples'
    print ner_train_x.shape[1], ' train dimension'
    print ner_test_x.shape[0], ' test samples'

    #minibatch_size_ner = len(ner_train_x) / n_minibatches
    minibatch_size_ner = 128



    # ----- Train Model ----- #

    dev_scores_ner, test_scores_ner = Trainer.trainModel(model_ner, model_train_input_ner,
                                                                               ner_train_y_cat, number_of_epochs,
                                                                               minibatch_size_ner,
                                                                               model_dev_input_ner,
                                                                               ner_dev_y, model_test_input_ner,
                                                                               ner_test_y)
    return model_ner, dev_scores_ner, test_scores_ner

def getPOSModel(embeddings, word2Idx):
    # Read in files
    (pos_train_x, pos_train_case_x, pos_train_y, pos_train_y_cat), (pos_dev_x, pos_dev_case_x, pos_dev_y), (
        pos_test_x, pos_test_case_x, pos_test_y) = UDPos.readDataset(windowSize, word2Idx, caseLookup)
    pos_n_out = pos_train_y_cat.shape[1]

    model_train_input_pos = [pos_train_x, pos_train_case_x]
    model_dev_input_pos = [pos_dev_x, pos_dev_case_x]
    model_test_input_pos = [pos_test_x, pos_test_case_x]

    # ----- Build Model ----- #
    model_pos = POS.buildPosModel(n_in, embeddings, n_in_case, numHiddenUnitsPOS, pos_n_out, metrics=metrics)

    print pos_train_x.shape[0], ' train samples'
    print pos_train_x.shape[1], ' train dimension'
    print pos_test_x.shape[0], ' test samples'

    # minibatch_size_pos = len(pos_train_x) / n_minibatches
    minibatch_size_pos = 128

    # ----- Train Model ----- #
    result = Trainer.trainModel(model_pos, model_train_input_pos, pos_train_y_cat,
                                                             number_of_epochs, minibatch_size_pos, model_dev_input_pos,
                                                             pos_dev_y, model_test_input_pos, pos_test_y)

    return model_pos

def getPOSModelGivenInput(word2Idx, input_layers, inputs):
    # Read in files
    (pos_train_x, pos_train_case_x, pos_train_y, pos_train_y_cat), (pos_dev_x, pos_dev_case_x, pos_dev_y), (
        pos_test_x, pos_test_case_x, pos_test_y) = UDPos.readDataset(windowSize, word2Idx, caseLookup)
    pos_n_out = pos_train_y_cat.shape[1]

    model_train_input_pos = [pos_train_x, pos_train_case_x]
    model_dev_input_pos = [pos_dev_x, pos_dev_case_x]
    model_test_input_pos = [pos_test_x, pos_test_case_x]

    # ----- Build Model ----- #
    model_pos = POS.buildPosModelGivenInput(input_layers, inputs, numHiddenUnitsPOS, pos_n_out)

    print pos_train_x.shape[0], ' train samples'
    print pos_train_x.shape[1], ' train dimension'
    print pos_test_x.shape[0], ' test samples'

    # minibatch_size_pos = len(pos_train_x) / n_minibatches
    minibatch_size_pos = 128

    # ----- Train Model ----- #
    result = Trainer.trainModel(model_pos, model_train_input_pos, pos_train_y_cat,
                                                             number_of_epochs, minibatch_size_pos, model_dev_input_pos,
                                                             pos_dev_y, model_test_input_pos, pos_test_y)

    return model_pos