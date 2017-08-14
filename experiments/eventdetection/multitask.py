from embeddings.dependency_based_word_embeddings import DependencyBasedWordEmbeddings as Embeddings
from datasets.wsj_pos import WSJPos
from datasets.universal_dependencies_pos import UDPos
from datasets.conll_ner import CoNLLNer
from datasets.conll_chunking import CoNLLChunking
from datasets.ace_ed import ACEED
from datasets.tac2015_ed import TACED
from datasets.tempeval3_ed import TempevalED
from datasets.ecbplus_ed import ECBPlusED
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


def getDataForConfig(params, config):
    datasets = []
    for task in config:
        if 'pos' == task:
            data = datasetFormat('wsj_pos_',
                                 WSJPos.readDataset(params['window_size'],
                                                    word2Idx, case2Idx))
            data['measurer'] = Measurer.measureAccuracy
            datasets.append(data)

        elif 'ner' == task:
            data = datasetFormat('ner_',
                                 CoNLLNer.readDataset(params['window_size'],
                                                      word2Idx, case2Idx))
            data['measurer'] = Measurer.create_compute_BIOf1(
                data['dicts'][-1])  # idx2label is last element in dicts
            datasets.append(data)

        elif 'chunking' == task:
            data = datasetFormat('chunking_',
                                 CoNLLChunking.readDataset(
                                     params['window_size'], word2Idx,
                                     case2Idx))
            data['measurer'] = Measurer.create_compute_BIOf1(
                data['dicts'][-1])  # idx2label is last element in dicts
            datasets.append(data)

        elif 'ace' == task:
            data = datasetFormat('ace_',
                                 ACEED.readDataset(params['window_size'],
                                                   word2Idx, case2Idx))
            data['measurer'] = Measurer.create_compute_BIOf1(
                data['dicts'][-1])  # idx2label is last element in dicts
            datasets.append(data)

        elif 'ecb' == task:
            data = datasetFormat('ecb_',
                                 ECBPlusED.readDataset(params['window_size'],
                                                       word2Idx, case2Idx))
            data['measurer'] = Measurer.create_compute_BIOf1(
                data['dicts'][-1])  # idx2label is last element in dicts
            datasets.append(data)

        elif 'tac' == task:
            data = datasetFormat('tac_',
                                 TACED.readDataset(params['window_size'],
                                                   word2Idx, case2Idx))
            data['measurer'] = Measurer.create_compute_BIOf1(
                data['dicts'][-1])  # idx2label is last element in dicts
            datasets.append(data)

        elif 'tempeval' == task:
            data = datasetFormat('tempeval_',
                                 TempevalED.readDataset(
                                     params['window_size'], word2Idx,
                                     case2Idx))
            data['measurer'] = Measurer.create_compute_BIOf1(
                data['dicts'][-1])  # idx2label is last element in dicts
            datasets.append(data)
    return datasets


def datasetFormat(name, dataset):
    [input_train, train_y_cat], [input_dev, dev_y], [input_test,
                                                     test_y], dicts = dataset
    return {
        'name': name,
        'train': {
            'input': input_train,
            'y': train_y_cat
        },
        'dev': {
            'input': input_dev,
            'y': dev_y
        },
        'test': {
            'input': input_test,
            'y': test_y
        },
        'dicts': dicts
    }


def buildAndTrainMultiTaskModel(learning_params, config=[]):
    params = learning_params

    datasets = getDataForConfig(params, config)
    # calculate input dimensions
    # take the first dataset for it, because model input looks the same for all
    first_dataset = datasets[0]
    train_input = first_dataset['train']['input']
    n_in_x = train_input[0].shape[1]  # train_x is first element of train input
    n_in_case = train_input[1].shape[
        1]  # train_case is the second element of train input

    # ----- Build Model ----- #
    input_layers, inputs = InputBuilder.buildStandardModelInput(
        embeddings, case2Idx, n_in_x, n_in_case,
        params['update_word_embeddings'])
    model_info = map(lambda data: (data['name'], data['train']['y'].shape[1]),
                     datasets)

    models = Senna.buildMultiTaskModelGivenInput(input_layers, inputs, params,
                                                 model_info)

    # ----- Train Model ----- #
    return Trainer.trainMultiTaskModelWithIncreasingData(
        models, datasets, number_of_epochs, params['batch_size'])

def run_baseline_exp_with_fixed_params():
    fixed_params = {
        'update_word_embeddings': False,
        'window_size': 3,
        'batch_size': 32,
        'hidden_dims': 500,
        'activation': 'tanh',
        'dropout': 0.3,
        'optimizer': 'adam',
        'number_of_epochs': config.number_of_epochs
    }
    max_evals = config.number_of_evals
    for model_nr in range(max_evals):
        print "Model nr. ", model_nr
        print fixed_params

        if 'ace' in config.tasks:
            #runAceExp(fixed_params, ['ecb', 'tac', 'tempeval'])
            #runAceExp(fixed_params, ['ecb', 'tac'])
            #runAceExp(fixed_params, ['tac', 'tempeval'])
            runAceExp(fixed_params, ['ecb', 'tempeval'])
            runAceExp(fixed_params, ['ecb'])
            runAceExp(fixed_params, ['tac'])
            runAceExp(fixed_params, ['tempeval'])

        if 'tac' in config.tasks:
            #runTacExp(fixed_params, ['ace', 'ecb', 'tempeval'])
            #runTacExp(fixed_params, ['ace', 'ecb'])
            #runTacExp(fixed_params, ['ace', 'tempeval'])
            runTacExp(fixed_params, ['ecb', 'tempeval'])
            runTacExp(fixed_params, ['ecb'])
            runTacExp(fixed_params, ['ace'])
            runTacExp(fixed_params, ['tempeval'])

        if 'tempeval' in config.tasks:
            '''runTempevalExp(fixed_params, ['ace', 'ecb', 'tac'])
            runTempevalExp(fixed_params, ['ace', 'ecb'])
            runTempevalExp(fixed_params, ['ace', 'tac'])
            runTempevalExp(fixed_params, ['ecb', 'tac'])
            runTempevalExp(fixed_params, ['ace'])
            runTempevalExp(fixed_params, ['tac'])'''
            runTempevalExp(fixed_params, ['ecb'])

        if 'ecb' in config.tasks:
            '''runEcbExp(fixed_params, ['ace', 'tac', 'tempeval'])
            runEcbExp(fixed_params, ['ace', 'tac'])
            runEcbExp(fixed_params, ['tac', 'tempeval'])
            runEcbExp(fixed_params, ['ace', 'tempeval'])
            runEcbExp(fixed_params, ['ace'])
            runEcbExp(fixed_params, ['tac'])'''
            runEcbExp(fixed_params, ['tempeval'])


def runAceExp(params, config):
    ExperimentHelper.run_build_model(
        'ace',
        'multitask',
        params,
        buildAndTrainMultiTaskModel,
        'f1',
        transfer_config=['ace'] + config)


def runEcbExp(params, config):
    ExperimentHelper.run_build_model(
        'ecb',
        'multitask',
        params,
        buildAndTrainMultiTaskModel,
        'f1',
        transfer_config=['ecb'] + config)


def runTacExp(params, config):
    ExperimentHelper.run_build_model(
        'tac',
        'multitask',
        params,
        buildAndTrainMultiTaskModel,
        'f1',
        transfer_config=['tac'] + config)


def runTempevalExp(params, config):
    ExperimentHelper.run_build_model(
        'tempeval',
        'multitask',
        params,
        buildAndTrainMultiTaskModel,
        'f1',
        transfer_config=['tempeval'] + config)


run_baseline_exp_with_fixed_params()
