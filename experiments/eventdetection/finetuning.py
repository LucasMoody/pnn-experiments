from embeddings.dependency_based_word_embeddings import DependencyBasedWordEmbeddings as Embeddings
from datasets.ace_ed import ACEED
from datasets.ecbplus_ed import ECBPlusED
from datasets.tac2015_ed import TACED
from datasets.tempeval3_ed import TempevalED
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


def buildAndTrainFinetuningModel(reader,
                                 name_prefix='',
                                 learning_params=None,
                                 config=[]):
    params = learning_params

    print 'Finetuning config:', config

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

    model = Senna.buildModelWithFinetuning(
        input_layers,
        inputs,
        params,
        n_out,
        transfer_models[0],
        name_prefix)

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(idx2Label)
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
        measurer=biof1)

def buildAndTrainAceModel(learning_params=None, config=[]):
    return buildAndTrainFinetuningModel(
        ACEED.readDataset,
        'ace_',
        learning_params=learning_params,
        config=config)


def buildAndTrainEcbModel(learning_params=None, config=[]):
    return buildAndTrainFinetuningModel(
        ECBPlusED.readDataset,
        'ecb_',
        learning_params=learning_params,
        config=config)


def buildAndTrainTacModel(learning_params=None, config=[]):
    return buildAndTrainFinetuningModel(
        TACED.readDataset,
        'tac_',
        learning_params=learning_params,
        config=config)


def buildAndTrainTempevalModel(learning_params=None, config=[]):
    return buildAndTrainFinetuningModel(
        TempevalED.readDataset,
        'tempeval_',
        learning_params=learning_params,
        config=config)


def run_finetuning_exp_with_fixed_params():
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

        if 'ace' in config.tasks:
            runAceExp(fixed_params, ['ecb'])
            runAceExp(fixed_params, ['tac'])
            runAceExp(fixed_params, ['tempeval'])

        if 'tac' in config.tasks:
            runTacExp(fixed_params, ['ace'])
            runTacExp(fixed_params, ['ecb'])
            runTacExp(fixed_params, ['tempeval'])

        if 'tempeval' in config.tasks:
            #runTempevalExp(fixed_params, ['ace'])
            runTempevalExp(fixed_params, ['ecb'])
            #runTempevalExp(fixed_params, ['tac'])

        if 'ecb' in config.tasks:
            #runEcbExp(fixed_params, ['ace'])
            #runEcbExp(fixed_params, ['tac'])
            runEcbExp(fixed_params, ['tempeval'])

def runAceExp(params, config):
    ExperimentHelper.run_build_model(
        'ace',
        'finetuning',
        params,
        buildAndTrainAceModel,
        'f1',
        transfer_config=config)


def runEcbExp(params, config):
    ExperimentHelper.run_build_model(
        'ecb',
        'finetuning',
        params,
        buildAndTrainEcbModel,
        'f1',
        transfer_config=config)


def runTacExp(params, config):
    ExperimentHelper.run_build_model(
        'tac',
        'finetuning',
        params,
         buildAndTrainTacModel,
        'f1',
        transfer_config=config)


def runTempevalExp(params, config):
    ExperimentHelper.run_build_model(
        'tempeval',
        'finetuning',
        params,
        buildAndTrainTempevalModel,
        'f1',
        transfer_config=config)

run_finetuning_exp_with_fixed_params()
