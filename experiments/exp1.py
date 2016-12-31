from embeddings.dependency_based_word_embeddings import DependencyBasedWordEmbeddings as Embeddings
from datasets.conll_ner import CoNLLNer
from datasets.universal_dependencies_pos import UDPos
from models.NER import SennaNER as NER
from models.POS import SennaPOS as POS
from models import Trainer, InputBuilder
from measurements import Measurer
import random
from parameters import parameter_space
from logs import Logger
import config

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

number_of_epochs = config.number_of_epochs


'''windowSize = 3 # n to the left, n to the right
n_in = 2 * windowSize + 1
numHiddenUnitsPOS = 100
numHiddenUnitsNER = 100
n_minibatches = 1000
number_of_epochs = 1
metrics = []'''

# ----- metric results -----#
metric_results = []

#Casing matrix
case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING':7}
n_in_case = len(case2Idx)

# Read in embeddings
embeddings = Embeddings.embeddings
word2Idx = Embeddings.word2Idx

def buildAndTrainNERModel(learning_params = None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    word2Idx = Embeddings.word2Idx
    # ----- NER ----- #

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
    dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model_ner, model_train_input_ner,
                                                                           ner_train_y_cat, number_of_epochs,
                                                                           params['batch_size'], model_dev_input_ner,
                                                                           ner_dev_y, model_test_input_ner, ner_test_y,
                                                                   measurements=[iof1])


    return dev_scores, test_scores

def buildAndTrainPOSModel(learning_params = None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    word2Idx = Embeddings.word2Idx
    # ----- NER ----- #

    [pos_input_train, pos_train_y_cat], [pos_input_dev, pos_dev_y], [pos_input_test, pos_test_y]= UDPos.readDataset(params['window_size'], word2Idx, case2Idx)
    [pos_train_x, pos_train_case_x] = pos_input_train
    [pos_dev_x, pos_dev_case_x] = pos_input_dev
    [pos_test_x, pos_test_case_x] = pos_input_test
    pos_n_out = pos_train_y_cat.shape[1]

    model_train_input_pos = [pos_train_x, pos_train_case_x]
    model_dev_input_pos = [pos_dev_x, pos_dev_case_x]
    model_test_input_pos = [pos_test_x, pos_test_case_x]

    n_in_x = pos_train_x.shape[1]
    n_in_casing = pos_train_case_x.shape[1]

    # ----- Build Model ----- #
    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing)
    model_pos = POS.buildPosModelGivenInput(input_layers, inputs, params, pos_n_out)

    print pos_train_x.shape[0], ' train samples'
    print pos_train_x.shape[1], ' train dimension'
    print pos_test_x.shape[0], ' test samples'


    # ----- Train Model ----- #
    dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model_pos, model_train_input_pos,
                                                                           pos_train_y_cat, number_of_epochs,
                                                                           params['batch_size'], model_dev_input_pos,
                                                                           pos_dev_y, model_test_input_pos, pos_test_y,
                                                                   measurements=[Measurer.measureAccuracy])


    return dev_scores, test_scores

max_evals = config.number_of_evals

for model_nr in xrange(max_evals):
    params = {}
    for key, values in parameter_space.space.iteritems():
        params[key] = random.choice(values)

    print "Model nr. ", model_nr
    best_dev_scores_ner, best_test_scores_ner = buildAndTrainNERModel(params)
    best_dev_scores_pos, best_test_scores_pos = buildAndTrainPOSModel(params)
    print params
    for (sample_scores, sample) in best_dev_scores_ner:
        for score in sample_scores:
            print "Max acc dev ner: %.4f in epoch with %d samples: %d" % (score[0][2], sample, score[1])
            Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_1', 'ner', 'dev', params, score[0][2], score[1], sample)
    for (sample_scores, sample) in best_test_scores_ner:
        for score in sample_scores:
            print "Max acc test ner: %.4f in epoch with %d samples: %d" % (score[0][2], sample, score[1])
            Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_1', 'ner', 'test', params, score[0][2], score[1], sample)
    for (sample_scores, sample) in best_dev_scores_pos:
        for score in sample_scores:
            print "Max acc test pos: %.4f in epoch with %d samples: %d" % (score[0], sample, score[1])
            Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_1', 'pos', 'dev', params, score[0], score[1], sample)
    for (sample_scores, sample) in best_test_scores_pos:
        for score in sample_scores:
            print "Max acc test pos: %.4f in epoch with %d samples: %d" % (score[0], sample, score[1])
            Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_1', 'pos', 'test', params, score[0],
                                                 score[1], sample)