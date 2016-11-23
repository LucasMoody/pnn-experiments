import datasets.universal_dependencies_pos.UDPos as UDPos
import datasets.conll_ner.CoNLLNer as CoNLLNer
import embeddings.dependency_based_word_embeddings.DependencyBasedWordEmbeddings as Embeddings
from transfer import Extender
import numpy as np
import models.POS.SennaPOS as POS
import models.NER.SennaNER as NER
import models.Trainer as Trainer
from keras.layers import Input
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

# Read in embeddings
embeddings, word2Idx = Embeddings.readEmbeddings()


def buildAndTrainNERModel(transfer=False, additional_models_for_input = [], useModelInput=False, useHiddenWeights=False):
    # ----- NER ----- #

    (ner_train_x, ner_train_case_x, ner_train_y, ner_train_y_cat), (ner_dev_x, ner_dev_case_x, ner_dev_y), (
        ner_test_x, ner_test_case_x, ner_test_y) = CoNLLNer.readDataset(windowSize, word2Idx, caseLookup)
    ner_n_out = ner_train_y_cat.shape[1]

    model_train_input_ner = [ner_train_x, ner_train_case_x]
    model_dev_input_ner = [ner_dev_x, ner_dev_case_x]
    model_test_input_ner = [ner_test_x, ner_test_case_x]

    # exp2
    if(useModelInput):
        Extender.extendInputs(
            additional_models_for_input, model_train_input_ner, model_dev_input_ner, model_test_input_ner)

    # ----- Build Model ----- #
    model_ner = NER.buildNERModel(n_in, embeddings, n_in_case, numHiddenUnitsNER, ner_n_out, metrics=metrics,
                                  additional_models_for_input=additional_models_for_input, useModelInput=useModelInput, useHiddenWeights=useHiddenWeights)

    print ner_train_x.shape[0], ' train samples'
    print ner_train_x.shape[1], ' train dimension'
    print ner_test_x.shape[0], ' test samples'

    print "NER Hidden weights sum (before Train): ", np.sum(model_ner.get_layer(name='ner_hidden').W.get_value())

    #minibatch_size_ner = len(ner_train_x) / n_minibatches
    minibatch_size_ner = 128



    # ----- Train Model ----- #
    if(transfer):
        dev_scores_ner, test_scores_ner = Trainer.trainModelWithIncreasingData(model_ner, model_train_input_ner,
                                                                           ner_train_y_cat, number_of_epochs,
                                                                           minibatch_size_ner, model_dev_input_ner,
                                                                           ner_dev_y, model_test_input_ner, ner_test_y)
    else:
        dev_scores_ner, test_scores_ner = Trainer.trainModel(model_ner, model_train_input_ner,
                                                                               ner_train_y_cat, number_of_epochs,
                                                                               minibatch_size_ner,
                                                             model_dev_input_ner,
                                                                               ner_dev_y, model_test_input_ner,
                                                                               ner_test_y)

    print "NER Hidden weights sum (after NER train): ", np.sum(model_ner.get_layer(name='ner_hidden').W.get_value())

    return model_ner, dev_scores_ner, test_scores_ner


def buildAndTrainPOSModel(transfer=False, additional_models_for_input = [], useModelInput=False, useHiddenWeights=False):
    # Read in files
    (pos_train_x, pos_train_case_x, pos_train_y, pos_train_y_cat), (pos_dev_x, pos_dev_case_x, pos_dev_y), (
        pos_test_x, pos_test_case_x, pos_test_y) = UDPos.readDataset(windowSize, word2Idx, caseLookup)
    pos_n_out = pos_train_y_cat.shape[1]

    model_train_input_pos = [pos_train_x, pos_train_case_x]
    model_dev_input_pos = [pos_dev_x, pos_dev_case_x]
    model_test_input_pos = [pos_test_x, pos_test_case_x]

    #exp2
    if (useModelInput):
        Extender.extendInputs(additional_models_for_input, model_train_input_pos, model_dev_input_pos,
                              model_test_input_pos)

    # ----- Build Model ----- #
    model_pos = POS.buildPosModel(n_in, embeddings, n_in_case, numHiddenUnitsPOS, pos_n_out, metrics=metrics,
                                  additional_models_for_input=additional_models_for_input, useModelInput=useModelInput, useHiddenWeights=useHiddenWeights)

    print pos_train_x.shape[0], ' train samples'
    print pos_train_x.shape[1], ' train dimension'
    print pos_test_x.shape[0], ' test samples'
    print "POS Hidden weights sum (before Train): ", np.sum(model_pos.get_layer(name='pos_hidden').W.get_value())

    # minibatch_size_pos = len(pos_train_x) / n_minibatches
    minibatch_size_pos = 128

    # ----- Train Model ----- #
    if(transfer):
        dev_scores_pos, test_scores_pos = Trainer.trainModelWithIncreasingData(model_pos, model_train_input_pos, pos_train_y_cat,
                                                                           number_of_epochs,
                                                                           minibatch_size_pos,
                                                                               model_dev_input_pos, pos_dev_y,
                                                                               model_test_input_pos, pos_test_y)
    else:
        dev_scores_pos, test_scores_pos = Trainer.trainModel(model_pos, model_train_input_pos, pos_train_y_cat,
                                                             number_of_epochs, minibatch_size_pos, model_dev_input_pos,
                                                             pos_dev_y, model_test_input_pos, pos_test_y)

    print "POS Hidden weights sum (after POS train): ", np.sum(model_pos.get_layer(name='pos_hidden').W.get_value())

    return model_pos, dev_scores_pos, test_scores_pos


# ----- Plotting ----- #
# experiment 1
model_ner_1, dev_scores_ner_1, test_scores_ner_1 = buildAndTrainNERModel(transfer=True)
# add results for plotting
metric_results.append((dev_scores_ner_1, 'ner_dev_1'))
metric_results.append((test_scores_ner_1, 'ner_test_1'))

# experiment 2
model_pos, unused_dev_scores_pos, unused_test_scores_pos = buildAndTrainPOSModel()
model_ner_2, dev_scores_ner_2, test_scores_ner_2 = buildAndTrainNERModel(transfer=True, additional_models_for_input=[model_pos], useModelInput=True)
# add results for plotting
metric_results.append((dev_scores_ner_2, 'ner_dev_2'))
metric_results.append((test_scores_ner_2, 'ner_test_2'))

model_ner_3, dev_scores_ner_3, test_scores_ner_3 = buildAndTrainNERModel(transfer=True, additional_models_for_input=[model_pos], useHiddenWeights=True)
# add results for plotting
metric_results.append((dev_scores_ner_3, 'ner_dev_3'))
metric_results.append((test_scores_ner_3, 'ner_test_3'))

LearningCurve.plotLearningCurve(metric_results)

# experiment 1
model_pos_1, dev_scores_pos_1, test_scores_pos_1 = buildAndTrainPOSModel(transfer=True)
# add results for plotting
metric_results.append((dev_scores_pos_1, 'pos_dev_1'))
metric_results.append((test_scores_pos_1, 'pos_test_1'))

# experiment 2
model_ner, unused_dev_scores_ner, unused_test_scores_ner = buildAndTrainNERModel()
model_pos_2, dev_scores_pos_2, test_scores_pos_2 = buildAndTrainPOSModel(transfer=True, additional_models_for_input=[model_ner], useModelInput=True)
# add results for plotting
metric_results.append((dev_scores_pos_2, 'pos_dev_2'))
metric_results.append((test_scores_pos_2, 'pos_test_2'))

model_pos_3, dev_scores_pos_3, test_scores_pos_3 = buildAndTrainPOSModel(transfer=True, additional_models_for_input=[model_ner], useHiddenWeights=True)
# add results for plotting
metric_results.append((dev_scores_pos_3, 'pos_dev_3'))
metric_results.append((test_scores_pos_3, 'pos_test_3'))

LearningCurve.plotLearningCurve(metric_results)
