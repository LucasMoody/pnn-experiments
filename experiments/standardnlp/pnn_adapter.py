from embeddings.dependency_based_word_embeddings import DependencyBasedWordEmbeddings as Embeddings
from datasets.universal_dependencies_pos import UDPos
from datasets.wsj_pos import WSJPos
from datasets.conll_chunking import CoNLLChunking
from datasets.conll_ner import CoNLLNer
from models import Trainer, Senna
from models import InputBuilder
from transfer import TransferModelBuilder
from measurements import Measurer
import config
from experiments import ExperimentHelper
from parameters import parameter_space
import random

number_of_epochs = config.number_of_epochs

# ----- metric results -----#

# Casing matrix
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


def buildAndTrainPNNModel(reader,
                          measurer_creator,
                          name_prefix='',
                          learning_params=None,
                          config=[]):
    params = learning_params

    print 'PNN config:', config

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

    input_layers, inputs = InputBuilder.buildStandardModelInput(
        embeddings, case2Idx, n_in_x, n_in_casing,
        params['update_word_embeddings'])

    transfer_models = TransferModelBuilder.buildTransferModels(
        input_layers, inputs, params=params, config=config)

    model = Senna.buildModelWithAdapterPNN(
        input_layers,
        inputs,
        params,
        n_out,
        additional_models=transfer_models,
        name_prefix=name_prefix)

    # ----- Train Model ----- #
    measurer = measurer_creator(idx2Label)
    return Trainer.trainModelWithIncreasingData(
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

def buildAndTrainWSJPosModel(learning_params=None, config=[]):
    return buildAndTrainPNNModel(
        WSJPos.readDataset,
        lambda _: Measurer.measureAccuracy,
        name_prefix='wsj_pos_',
        learning_params=learning_params,
        config=config)


def buildAndTrainUDPosModel(learning_params=None, config=[]):
    return buildAndTrainPNNModel(
        UDPos.readDataset,
        lambda _: Measurer.measureAccuracy,
        name_prefix='ud_pos_',
        learning_params=learning_params,
        config=config)


def buildAndTrainNERModel(learning_params=None, config=[]):
    return buildAndTrainPNNModel(
        CoNLLNer.readDataset,
        Measurer.create_compute_BIOf1,
        name_prefix='ner_',
        learning_params=learning_params,
        config = config)


def buildAndTrainChunkingEDModel(learning_params=None, config=[]):
    return buildAndTrainPNNModel(
        CoNLLChunking.readDataset,
        Measurer.create_compute_BIOf1,
        name_prefix='chunking_',
        learning_params=learning_params,
        config=config)


def run_pnn_exp_with_fixed_params():
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

    for model_nr in xrange(max_evals):
        print "Model nr. ", model_nr

        if 'wsj_pos' in config.tasks:
            runWSJPosExp(fixed_params, ['ner', 'chunking'])
            runWSJPosExp(fixed_params, ['ner'])
            runWSJPosExp(fixed_params, ['chunking'])
            runWSJPosExp(fixed_params, ['ud_pos'])

        if 'ner' in config.tasks:
            runNerExp(fixed_params, ['pos', 'chunking'])
            runNerExp(fixed_params, ['pos'])
            runNerExp(fixed_params, ['chunking'])

        if 'chunking' in config.tasks:
            runChunkingExp(fixed_params, ['ner', 'pos'])
            runChunkingExp(fixed_params, ['pos'])
            runChunkingExp(fixed_params, ['ner'])

        if 'ud_pos' in config.tasks:
            runUDPosExp(fixed_params, ['pos'])

def runWSJPosExp(params, config):
    ExperimentHelper.run_build_model(
        'wsj_pos',
        'pnn_adapter',
        params,
        buildAndTrainWSJPosModel,
        'acc',
        transfer_config=config)


def runUDPosExp(params, config):
    ExperimentHelper.run_build_model(
        'ud_pos',
        'pnn_adapter',
        params,
        buildAndTrainUDPosModel,
        'acc',
        transfer_config=config)


def runNerExp(params, config):
    ExperimentHelper.run_build_model(
        'ner',
        'pnn_adapter',
        params,
        buildAndTrainNERModel,
        'f1',
        transfer_config=config)


def runChunkingExp(params, config):
    ExperimentHelper.run_build_model(
        'chunking',
        'pnn_adapter',
        params,
        buildAndTrainChunkingEDModel,
        'f1',
        transfer_config=config)

run_pnn_exp_with_fixed_params()
