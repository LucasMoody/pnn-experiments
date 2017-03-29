from embeddings.dependency_based_word_embeddings import DependencyBasedWordEmbeddings as Embeddings
from datasets.ace_ed import ACEED
from datasets.tac2015_ed import TACED
from datasets.tempeval3_ed import TempevalED
from datasets.ecbplus_ed import ECBPlusED
from models import Trainer, InputBuilder, Senna
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
        'activation': 'relu',
        'dropout': 0.3,
        'optimizer': 'adam'
    }

number_of_epochs = config.number_of_epochs

# ----- metric results -----#
metric_results = []

#Casing matrix
case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING':7}
n_in_case = len(case2Idx)

# Read in embeddings
embeddings = Embeddings.embeddings
word2Idx = Embeddings.word2Idx

def buildBaselineModel(reader, name_prefix='', learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = reader(params['window_size'], word2Idx, case2Idx)
    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, caseLookup, label2Idx, idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    # ----- Build Model ----- #
    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing, params['update_word_embeddings'])
    model = Senna.buildModelGivenInput(input_layers, inputs, params, n_out, name_prefix=name_prefix)

    print train_x.shape[0], ' train samples'
    print train_x.shape[1], ' train dimension'
    print test_x.shape[0], ' test samples'

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, input_train, train_y_cat, number_of_epochs,
                                                                           params['batch_size'], input_dev,
                                                                           dev_y, input_test, test_y, measurements=[biof1])


    return train_scores, dev_scores, test_scores

def buildAndTrainAceEDModel(learning_params = None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    # ----- NER ----- #
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = ACEED.readDataset(params['window_size'], word2Idx, case2Idx)
    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, caseLookup, label2Idx, idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    # ----- Build Model ----- #
    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing, params['update_word_embeddings'])
    model = Senna.buildModelGivenInput(input_layers, inputs, params, n_out, name_prefix='ace_ed_')

    print train_x.shape[0], ' train samples'
    print train_x.shape[1], ' train dimension'
    print test_x.shape[0], ' test samples'

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, input_train, train_y_cat, number_of_epochs,
                                                                           params['batch_size'], input_dev,
                                                                           dev_y, input_test, test_y, measurements=[biof1])


    return train_scores, dev_scores, test_scores

def buildAndTrainTacEDModel(learning_params = None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    # ----- NER ----- #
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = TACED.readDataset(params['window_size'], word2Idx, case2Idx)
    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, caseLookup, label2Idx, idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    # ----- Build Model ----- #
    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing, params['update_word_embeddings'])
    model = Senna.buildModelGivenInput(input_layers, inputs, params, n_out, name_prefix='tac_ed_')

    print train_x.shape[0], ' train samples'
    print train_x.shape[1], ' train dimension'
    print test_x.shape[0], ' test samples'

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, input_train, train_y_cat, number_of_epochs,
                                                                           params['batch_size'], input_dev,
                                                                           dev_y, input_test, test_y, measurements=[biof1])


    return train_scores, dev_scores, test_scores

def buildAndTrainTempevalEDModel(learning_params = None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = TempevalED.readDataset(params['window_size'], word2Idx, case2Idx)
    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, caseLookup, label2Idx, idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    # ----- Build Model ----- #
    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing, params['update_word_embeddings'])
    model = Senna.buildModelGivenInput(input_layers, inputs, params, n_out, name_prefix='tac_ed_')

    print train_x.shape[0], ' train samples'
    print train_x.shape[1], ' train dimension'
    print test_x.shape[0], ' test samples'

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, input_train, train_y_cat, number_of_epochs,
                                                                           params['batch_size'], input_dev,
                                                                           dev_y, input_test, test_y, measurements=[biof1])


    return train_scores, dev_scores, test_scores

def buildAndTrainECBPlusEDModel(learning_params = None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    # ----- NER ----- #
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = ECBPlusED.readDataset(params['window_size'], word2Idx, case2Idx)
    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, caseLookup, label2Idx, idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    # ----- Build Model ----- #
    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing, params['update_word_embeddings'])
    model = Senna.buildModelGivenInput(input_layers, inputs, params, n_out, name_prefix='ecb_ed_')

    print train_x.shape[0], ' train samples'
    print train_x.shape[1], ' train dimension'
    print test_x.shape[0], ' test samples'

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, input_train, train_y_cat, number_of_epochs,
                                                                           params['batch_size'], input_dev,
                                                                           dev_y, input_test, test_y, measurements=[biof1])


    return train_scores, dev_scores, test_scores



def run_baseline_exp_with_fixed_params():
    fixed_params = {
        'update_word_embeddings': False,
        'window_size': 3,
        'batch_size': 32,
        'hidden_dims': 100,
        'activation': 'tanh',
        'dropout': 0.3,
        'optimizer': 'adam',
        'number_of_epochs': config.number_of_epochs
    }
    max_evals = config.number_of_evals
    for model_nr in range(max_evals):
        print "Model nr. ", model_nr
        print fixed_params

        if 'ace_ed' in config.tasks:
            run_build_model('ace_ed', 'baseline', fixed_params, buildAndTrainAceEDModel, 'f1', 'none')

        if 'tac_ed' in config.tasks:
            run_build_model('tac_ed', 'baseline', fixed_params, buildAndTrainTacEDModel, 'f1', 'none')

        if 'tempeval_ed' in config.tasks:
            run_build_model('tempeval_ed', 'baseline', fixed_params, buildAndTrainTempevalEDModel, 'f1', 'none')

        if 'ecb_ed' in config.tasks:
            run_build_model('ecb_ed', 'baseline', fixed_params, buildAndTrainTempevalEDModel, 'f1', 'none')



def run_build_model(task, exp, params, build_model_func, score_name, transfer_models):
    train_scores, dev_scores, test_scores = build_model_func(params)
    print params
    for (sample_scores, sample) in train_scores:
        for score in sample_scores:
            print "Max {0} train {1} with {2}: {3:.4f} in epoch: {4} with samples: {5}".format(score_name, task, transfer_models, score[0], score[1], sample)
            Logger.save_reduced_datasets_results(config.experiments_log_path, exp, task, 'train', params, score[0], score[1], sample, transfer_models)
    for (sample_scores, sample) in dev_scores:
        for score in sample_scores:
            print "Max {0} dev {1} with {2}: {3:.4f} in epoch: {4} with samples: {5}".format(score_name, task, transfer_models, score[0], score[1], sample)
            Logger.save_reduced_datasets_results(config.experiments_log_path, exp, task, 'dev', params, score[0], score[1], sample, transfer_models)
    for (sample_scores, sample) in test_scores:
        for score in sample_scores:
            print "Max {0} test {1} with {2}: {3:.4f} in epoch: {4} with samples: {5}".format(score_name, task, transfer_models, score[0], score[1], sample)
            Logger.save_reduced_datasets_results(config.experiments_log_path, exp, task, 'test', params, score[0], score[1], sample, transfer_models)

    print '\n\n-------------------- END --------------------\n\n'

run_baseline_exp_with_fixed_params()