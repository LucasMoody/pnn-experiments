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

def getDataForConfig(params, config):
    datasets = []
    if 'pos' in config:
        data = datasetFormat('wsj_pos_', WSJPos.readDataset(params['window_size'], word2Idx, case2Idx))
        data['measurer'] = Measurer.measureAccuracy
        datasets.append(data)

    if 'ner' in config:
        data = datasetFormat('ner_', CoNLLNer.readDataset(params['window_size'], word2Idx, case2Idx))
        data['measurer'] = Measurer.create_compute_BIOf1(data['dicts'][-1])  # idx2label is last element in dicts
        datasets.append(data)

    if 'chunking' in config:
        data = datasetFormat('chunking_', CoNLLChunking.readDataset(params['window_size'], word2Idx, case2Idx))
        data['measurer'] = Measurer.create_compute_BIOf1(data['dicts'][-1])  # idx2label is last element in dicts
        datasets.append(data)

    if 'ace' in config:
        data = datasetFormat('ace_', ACEED.readDataset(params['window_size'], word2Idx, case2Idx))
        data['measurer'] = Measurer.create_compute_BIOf1(data['dicts'][-1])  # idx2label is last element in dicts
        datasets.append(data)

    if 'ecb' in config:
        data = datasetFormat('ecb_', ECBPlusED.readDataset(params['window_size'], word2Idx, case2Idx))
        data['measurer'] = Measurer.create_compute_BIOf1(data['dicts'][-1])  # idx2label is last element in dicts
        datasets.append(data)

    if 'tac' in config:
        data = datasetFormat('tac_', TACED.readDataset(params['window_size'], word2Idx, case2Idx))
        data['measurer'] = Measurer.create_compute_BIOf1(data['dicts'][-1])  # idx2label is last element in dicts
        datasets.append(data)

    if 'tempeval' in config:
        data = datasetFormat('tempeval_', TempevalED.readDataset(params['window_size'], word2Idx, case2Idx))
        data['measurer'] = Measurer.create_compute_BIOf1(data['dicts'][-1])  # idx2label is last element in dicts
        datasets.append(data)
    return datasets

def datasetFormat(name, dataset):
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = dataset
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

def buildAndTrainMultiTaskModel(learning_params = None, config={}):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    datasets = getDataForConfig(params, config)
    # calculate input dimensions
    # take the first dataset for it, because model input looks the same for all
    first_dataset = datasets[0]
    train_input = first_dataset['train']['input']
    n_in_x = train_input[0].shape[1] # train_x is first element of train input
    n_in_case = train_input[1].shape[1] # train_case is the second element of train input

    # ----- Build Model ----- #
    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_case,
                                                                params['update_word_embeddings'])
    model_info = map(lambda data: (data['name'], data['train']['y'].shape[1]),datasets)

    models = Senna.buildMultiTaskModelGivenInput(input_layers, inputs, params, model_info)

    # ----- Train Model ----- #
    train_scores, dev_scores, test_scores = Trainer.trainMultiTaskModelWithIncreasingData(models, datasets, number_of_epochs, params['batch_size'])

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

        if 'ace' in config.tasks:
            ExperimentHelper.run_build_model('ace', 'multitask', fixed_params, buildAndTrainMultiTaskModel, 'f1', transfer_config={'ace', 'pos', 'ner', 'chunking', 'ecb', 'tac', 'tempeval'})
            ExperimentHelper.run_build_model('ace', 'multitask', fixed_params, buildAndTrainMultiTaskModel, 'f1', transfer_config={'ace', 'ecb', 'tac', 'tempeval'})
            ExperimentHelper.run_build_model('ace', 'multitask', fixed_params, buildAndTrainMultiTaskModel, 'f1', transfer_config={'ace', 'ecb', 'tac'})
            ExperimentHelper.run_build_model('ace', 'multitask', fixed_params, buildAndTrainMultiTaskModel, 'f1', transfer_config={'ace', 'ecb', 'tempeval'})
            ExperimentHelper.run_build_model('ace', 'multitask', fixed_params, buildAndTrainMultiTaskModel, 'f1', transfer_config={'ace', 'tac', 'tempeval'})
            ExperimentHelper.run_build_model('ace', 'multitask', fixed_params, buildAndTrainMultiTaskModel, 'f1', transfer_config={'ace', 'ecb'})
            ExperimentHelper.run_build_model('ace', 'multitask', fixed_params, buildAndTrainMultiTaskModel, 'f1', transfer_config={'ace', 'tac'})
            ExperimentHelper.run_build_model('ace', 'multitask', fixed_params, buildAndTrainMultiTaskModel, 'f1', transfer_config={'ace', 'tempeval'})

        if 'tac' in config.tasks:
            ExperimentHelper.run_build_model('tac', 'baseline', fixed_params, buildAndTrainMultiTaskModel, 'f1', 'none')

        if 'tempeval' in config.tasks:
            ExperimentHelper.run_build_model('tempeval', 'baseline', fixed_params, buildAndTrainMultiTaskModel, 'f1', 'none')

        if 'ecb' in config.tasks:
            ExperimentHelper.run_build_model('ecb', 'baseline', fixed_params, buildAndTrainMultiTaskModel, 'f1', 'none')

run_baseline_exp_with_fixed_params()