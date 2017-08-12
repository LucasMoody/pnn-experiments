from embeddings.dependency_based_word_embeddings import DependencyBasedWordEmbeddings as Embeddings
from datasets.universal_dependencies_pos import UDPos
from datasets.wsj_pos import WSJPos
from datasets.conll_chunking import CoNLLChunking
from datasets.conll_ner import CoNLLNer
from models import Trainer, InputBuilder, Senna
from measurements import Measurer
import random
from parameters import parameter_space
import config
from experiments import ExperimentHelper

number_of_epochs = config.number_of_epochs

# ----- metric results -----#
metric_results = []

#Casing matrix
case2Idx = {
    'numeric': 0,
    'allLower': 1,
    'allUpper': 2,
    'initialUpper': 3,
    'other': 4,
    'mainly_numeric': 5,
    'contains_digit': 6,
    'PADDING': 7
}
n_in_case = len(case2Idx)

# Read in embeddings
embeddings = Embeddings.embeddings
word2Idx = Embeddings.word2Idx


def buildBaselineModel(reader, measurer_creator, name_prefix='', learning_params=None):
    params = learning_params

    [input_train, train_y_cat], [input_dev,
                                 dev_y], [input_test, test_y], dicts = reader(
                                     params['window_size'], word2Idx, case2Idx)
    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, caseLookup, label2Idx, idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    # ----- Build Model ----- #
    input_layers, inputs = InputBuilder.buildStandardModelInput(
        embeddings, case2Idx, n_in_x, n_in_casing,
        params['update_word_embeddings'])
    model = Senna.buildModelGivenInput(
        input_layers, inputs, params, n_out, name_prefix=name_prefix)

    print train_x.shape[0], ' train samples'
    print train_x.shape[1], ' train dimension'
    print test_x.shape[0], ' test samples'

    # ----- Train Model ----- #
    measurer = measurer_creator(idx2Label)
    scores = Trainer.trainModelWithIncreasingData(
        model,
        input_train,
        train_y_cat,
        number_of_epochs,
        params['batch_size'],
        input_dev,
        dev_y,
        input_test,
        test_y,
        measurer=measurer)

    return scores


def buildAndTrainWSJPosModel(learning_params=None):
    return buildBaselineModel(
        WSJPos.readDataset,
        lambda _: Measurer.measureAccuracy,
        name_prefix='wsj_pos_',
        learning_params=learning_params)


def buildAndTrainUDPosModel(learning_params=None):
    return buildBaselineModel(
        UDPos.readDataset,
        lambda _: Measurer.measureAccuracy,
        name_prefix='ud_pos_',
        learning_params=learning_params)


def buildAndTrainNERModel(learning_params=None):
    return buildBaselineModel(
        CoNLLNer.readDataset,
        Measurer.create_compute_BIOf1,
        name_prefix='ner_',
        learning_params=learning_params)


def buildAndTrainChunkingEDModel(learning_params=None):
    return buildBaselineModel(
        CoNLLChunking.readDataset,
        Measurer.create_compute_BIOf1,
        name_prefix='chunking_',
        learning_params=learning_params)


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

        if 'wsj_pos' in config.tasks:
            ExperimentHelper.run_build_model('wsj_pos', 'baseline', fixed_params,
                                             buildAndTrainWSJPosModel, 'acc',
                                             'none')

        if 'ud_pos' in config.tasks:
            ExperimentHelper.run_build_model('ud_pos', 'baseline', fixed_params,
                                             buildAndTrainUDPosModel,
                                             'acc', 'none')

        if 'ner' in config.tasks:
            ExperimentHelper.run_build_model('ner', 'baseline', fixed_params,
                                             buildAndTrainNERModel, 'f1',
                                             'none')

        if 'chunking' in config.tasks:
            ExperimentHelper.run_build_model('chunking', 'baseline', fixed_params,
                                             buildAndTrainChunkingEDModel, 'f1', 'none')

run_baseline_exp_with_fixed_params()
