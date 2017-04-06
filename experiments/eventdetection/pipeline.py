from keras.layers import Input, Embedding, Flatten, merge

import embeddings.dependency_based_word_embeddings.DependencyBasedWordEmbeddings as Embeddings
from models import Trainer, InputBuilder
from datasets.ace_ed import ACEED
from datasets.tac2015_ed import TACED
from datasets.ecbplus_ed import ECBPlusED
from datasets.tempeval3_ed import TempevalED
from datasets.wsj_pos import WSJPos
from datasets.conll_ner import CoNLLNer
from datasets.conll_chunking import CoNLLChunking
from models import Senna
from optimizer import OptimizedModels
from parameters import parameter_space
from measurements import Measurer
import config
from experiments import ExperimentHelper

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

best_tac_window_size = 3
best_tempeval_window_size = 3
best_ace_window_size = 3
best_ecb_window_size = 3
best_pos_window_size = 3
best_ner_window_size = 3
best_chunking_window_size = 3

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


def extendHelper(reader, word2Idx, case2Idx, best_window_size,
                 pipeline_model_builder, pipeline_dict):
    # ----- read Data with best window ----- #
    [input_train, events_train_y_cat], [input_dev, events_dev_y], [
        input_test, events_test_y
    ], dicts = reader(best_window_size, word2Idx, case2Idx)
    # calculate dims for model building
    [train_x, train_case_x] = input_train
    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    input_layers, inputs = InputBuilder.buildStandardModelInput(
        embeddings, case2Idx, n_in_x, n_in_casing)
    model_pipeline = pipeline_model_builder(input_layers, inputs)

    # predict data
    pred_train = model_pipeline.predict(input_train, verbose=0).argmax(axis=-1)
    pred_dev = model_pipeline.predict(input_dev, verbose=0).argmax(axis=-1)
    pred_test = model_pipeline.predict(input_test, verbose=0).argmax(axis=-1)

    #
    label2Idx, idx2Label = pipeline_dict
    pred_train_labels = map(lambda idx: idx2Label[idx], pred_train)
    pred_dev_labels = map(lambda idx: idx2Label[idx], pred_dev)
    pred_test_labels = map(lambda idx: idx2Label[idx], pred_test)

    return pred_train_labels, pred_dev_labels, pred_test_labels


def extendAceED():
    extendED(['pos', 'ner', 'chunking', 'ecb', 'tac', 'tempeval'], ACEED)


def extendEcbED():
    extendED(['pos', 'ner', 'chunking', 'ace', 'tac', 'tempeval'], ECBPlusED)


def extendTacED():
    extendED(['pos', 'ner', 'chunking', 'ace', 'ecb', 'tempeval'], TACED)


def extendTempevalED():
    extendED(['pos', 'ner', 'chunking', 'ace', 'ecb', 'tac'], TempevalED)


def extendED(config, dataset):
    train_extensions = []
    dev_extensions = []
    test_extensions = []

    for ds in config:

        if 'pos' == ds:
            # ----- labels from pos  ----- #
            pred_train_labels, pred_dev_labels, pred_test_labels = extendHelper(
                dataset.readDataset, word2Idx, case2Idx,
                best_pos_window_size, OptimizedModels.getWSJPOSModelGivenInput,
                WSJPos.getLabelDict())
            train_extensions.append(pred_train_labels)
            dev_extensions.append(pred_dev_labels)
            test_extensions.append(pred_test_labels)

        elif 'ner' == ds:
            # ----- labels from ner  ----- #
            pred_train_labels, pred_dev_labels, pred_test_labels = extendHelper(
                dataset.readDataset, word2Idx, case2Idx, best_ner_window_size,
                OptimizedModels.getNERModelGivenInput, CoNLLNer.getLabelDict())
            train_extensions.append(pred_train_labels)
            dev_extensions.append(pred_dev_labels)
            test_extensions.append(pred_test_labels)

        elif 'chunking' == ds:
            # ----- labels from chunking  ----- #
            pred_train_labels, pred_dev_labels, pred_test_labels = extendHelper(
                dataset.readDataset, word2Idx, case2Idx,
                best_chunking_window_size,
                OptimizedModels.getChunkingModelGivenInput,
                CoNLLChunking.getLabelDict())
            train_extensions.append(pred_train_labels)
            dev_extensions.append(pred_dev_labels)
            test_extensions.append(pred_test_labels)

        elif 'ace' == ds:
            # ----- labels from ace ----- #
            pred_train_labels, pred_dev_labels, pred_test_labels = extendHelper(
                dataset.readDataset, word2Idx, case2Idx, best_ace_window_size,
                OptimizedModels.getAceEDModelGivenInput, ACEED.getLabelDict())
            train_extensions.append(pred_train_labels)
            dev_extensions.append(pred_dev_labels)
            test_extensions.append(pred_test_labels)

        elif 'ecb' == ds:
            # ----- labels from ecb  ----- #
            pred_train_labels, pred_dev_labels, pred_test_labels = extendHelper(
                dataset.readDataset, word2Idx, case2Idx, best_ecb_window_size,
                OptimizedModels.getEcbEDModelGivenInput,
                ECBPlusED.getLabelDict())
            train_extensions.append(pred_train_labels)
            dev_extensions.append(pred_dev_labels)
            test_extensions.append(pred_test_labels)

        elif 'tac' == ds:
            # ----- labels from tac  ----- #
            pred_train_labels, pred_dev_labels, pred_test_labels = extendHelper(
                dataset.readDataset, word2Idx, case2Idx, best_tac_window_size,
                OptimizedModels.getTacEDModelGivenInput, TACED.getLabelDict())
            train_extensions.append(pred_train_labels)
            dev_extensions.append(pred_dev_labels)
            test_extensions.append(pred_test_labels)

        elif 'tempeval' == ds:
            # ----- labels from tempeval  ----- #
            pred_train_labels, pred_dev_labels, pred_test_labels = extendHelper(
                dataset.readDataset, word2Idx, case2Idx,
                best_tempeval_window_size,
                OptimizedModels.getTempevalEDModelGivenInput,
                TempevalED.getLabelDict())
            train_extensions.append(pred_train_labels)
            dev_extensions.append(pred_dev_labels)
            test_extensions.append(pred_test_labels)

    dataset.extendDataset(train_extensions, dev_extensions, test_extensions)

    print '\n--------------------\n          DONE\n--------------------\n'


def buildAndTrainAceModel(learning_params=None, config=[]):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train,
     train_y_cat], [input_dev,
                    dev_y], [input_test, test_y], dicts = ACEED.readDatasetExt(
                        params['window_size'], word2Idx, case2Idx)

    [
        _, events_pos2Idx, events_ner2Idx, events_chunking2Idx, events_ecb2Idx,
        events_tac2Idx, events_tempeval2Idx, _, events_label2Idx,
        events_idx2Label
    ] = dicts

    [
        events_train_x, events_train_casing_x, events_train_pos_x,
        events_train_ner_x, events_train_chunking_x, events_train_ecb_x,
        events_train_tac_x, events_train_tempeval_x
    ] = input_train

    n_out = train_y_cat.shape[1]
    n_in_x = events_train_x.shape[1]
    n_in_pos = events_train_pos_x.shape[1]
    n_in_ner = events_train_ner_x.shape[1]
    n_in_chunking = events_train_chunking_x.shape[1]
    n_in_casing = events_train_casing_x.shape[1]
    n_in_tac = events_train_tac_x.shape[1]
    n_in_ecb = events_train_ecb_x.shape[1]
    n_in_tempeval = events_train_tempeval_x.shape[1]

    # prepare config
    config_values = {
        'words': {
            'n': n_in_x,
            'idx': embeddings,
            'pos': 0
        },
        'casing': {
            'n': n_in_casing,
            'idx': case2Idx,
            'pos': 1
        },
        'pos': {
            'n': n_in_pos,
            'idx': events_pos2Idx,
            'pos': 2
        },
        'ner': {
            'n': n_in_ner,
            'idx': events_ner2Idx,
            'pos': 3
        },
        'chunking': {
            'n': n_in_chunking,
            'idx': events_chunking2Idx,
            'pos': 4
        },
        'ecb': {
            'n': n_in_ecb,
            'idx': events_ecb2Idx,
            'pos': 5
        },
        'tac': {
            'n': n_in_tac,
            'idx': events_tac2Idx,
            'pos': 6
        },
        'tempeval': {
            'n': n_in_tempeval,
            'idx': events_tempeval2Idx,
            'pos': 7
        }
    }

    selectedFeatures = {key: config_values[key] for key in config}

    input_layers, inputs = InputBuilder.buildPipelineModelInput(
        selectedFeatures)

    input_layers_merged = merge(input_layers, mode='concat')

    model = Senna.buildModelGivenInput(
        input_layers_merged, inputs, params, n_out, name_prefix='ace_')

    # SELECT FEATURES WHICH APPEAR IN THE CONFIG
    indices = sorted(
        [selectedFeatures[feature]['pos'] for feature in selectedFeatures])
    model_train = [input_train[i] for i in indices]
    model_dev = [input_dev[i] for i in indices]
    model_test = [input_test[i] for i in indices]

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(events_idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(
        model,
        model_train,
        train_y_cat,
        number_of_epochs,
        params['batch_size'],
        model_dev,
        dev_y,
        model_test,
        test_y,
        measurer=biof1)

    return train_scores, dev_scores, test_scores


def buildAndTrainEcbModel(learning_params=None, config=[]):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train,
     train_y_cat], [input_dev,
                    dev_y], [input_test,
                             test_y], dicts = ECBPlusED.readDatasetExt(
                                 params['window_size'], word2Idx, case2Idx)

    [
        _, events_pos2Idx, events_ner2Idx, events_chunking2Idx, events_ace2Idx,
        events_tac2Idx, events_tempeval2Idx, _, events_label2Idx,
        events_idx2Label
    ] = dicts

    [
        events_train_x, events_train_casing_x, events_train_pos_x,
        events_train_ner_x, events_train_chunking_x, events_train_ace_x,
        events_train_tac_x, events_train_tempeval_x
    ] = input_train

    n_out = train_y_cat.shape[1]
    n_in_x = events_train_x.shape[1]
    n_in_pos = events_train_pos_x.shape[1]
    n_in_ner = events_train_ner_x.shape[1]
    n_in_chunking = events_train_chunking_x.shape[1]
    n_in_casing = events_train_casing_x.shape[1]
    n_in_tac = events_train_tac_x.shape[1]
    n_in_ace = events_train_ace_x.shape[1]
    n_in_tempeval = events_train_tempeval_x.shape[1]

    # prepare config
    config_values = {
        'words': {
            'n': n_in_x,
            'idx': embeddings,
            'pos': 0
        },
        'casing': {
            'n': n_in_casing,
            'idx': case2Idx,
            'pos': 1
        },
        'pos': {
            'n': n_in_pos,
            'idx': events_pos2Idx,
            'pos': 2
        },
        'ner': {
            'n': n_in_ner,
            'idx': events_ner2Idx,
            'pos': 3
        },
        'chunking': {
            'n': n_in_chunking,
            'idx': events_chunking2Idx,
            'pos': 4
        },
        'ace': {
            'n': n_in_ace,
            'idx': events_ace2Idx,
            'pos': 5
        },
        'tac': {
            'n': n_in_tac,
            'idx': events_tac2Idx,
            'pos': 6
        },
        'tempeval': {
            'n': n_in_tempeval,
            'idx': events_tempeval2Idx,
            'pos': 7
        }
    }

    selectedFeatures = {key: config_values[key] for key in config}

    input_layers, inputs = InputBuilder.buildPipelineModelInput(
        selectedFeatures)

    input_layers_merged = merge(input_layers, mode='concat')

    model = Senna.buildModelGivenInput(
        input_layers_merged, inputs, params, n_out, name_prefix='ace_')

    # SELECT FEATURES WHICH APPEAR IN THE CONFIG
    indices = sorted(
        [selectedFeatures[feature]['pos'] for feature in selectedFeatures])
    model_train = [input_train[i] for i in indices]
    model_dev = [input_dev[i] for i in indices]
    model_test = [input_test[i] for i in indices]

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(events_idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(
        model,
        model_train,
        train_y_cat,
        number_of_epochs,
        params['batch_size'],
        model_dev,
        dev_y,
        model_test,
        test_y,
        measurer=biof1)

    return train_scores, dev_scores, test_scores


def buildAndTrainTacModel(learning_params=None, config=[]):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train,
     train_y_cat], [input_dev,
                    dev_y], [input_test, test_y], dicts = TACED.readDatasetExt(
                        params['window_size'], word2Idx, case2Idx)

    [
        _, events_pos2Idx, events_ner2Idx, events_chunking2Idx, events_ace2Idx,
        events_ecb2Idx, events_tempeval2Idx, _, events_label2Idx,
        events_idx2Label
    ] = dicts

    [
        events_train_x, events_train_casing_x, events_train_pos_x,
        events_train_ner_x, events_train_chunking_x, events_train_ace_x,
        events_train_ecb_x, events_train_tempeval_x
    ] = input_train

    n_out = train_y_cat.shape[1]
    n_in_x = events_train_x.shape[1]
    n_in_pos = events_train_pos_x.shape[1]
    n_in_ner = events_train_ner_x.shape[1]
    n_in_chunking = events_train_chunking_x.shape[1]
    n_in_casing = events_train_casing_x.shape[1]
    n_in_ace = events_train_ace_x.shape[1]
    n_in_ecb = events_train_ecb_x.shape[1]
    n_in_tempeval = events_train_tempeval_x.shape[1]

    # prepare config
    config_values = {
        'words': {
            'n': n_in_x,
            'idx': embeddings,
            'pos': 0
        },
        'casing': {
            'n': n_in_casing,
            'idx': case2Idx,
            'pos': 1
        },
        'pos': {
            'n': n_in_pos,
            'idx': events_pos2Idx,
            'pos': 2
        },
        'ner': {
            'n': n_in_ner,
            'idx': events_ner2Idx,
            'pos': 3
        },
        'chunking': {
            'n': n_in_chunking,
            'idx': events_chunking2Idx,
            'pos': 4
        },
        'ace': {
            'n': n_in_ace,
            'idx': events_ace2Idx,
            'pos': 5
        },
        'ecb': {
            'n': n_in_ecb,
            'idx': events_ecb2Idx,
            'pos': 6
        },
        'tempeval': {
            'n': n_in_tempeval,
            'idx': events_tempeval2Idx,
            'pos': 7
        }
    }

    selectedFeatures = {key: config_values[key] for key in config}

    input_layers, inputs = InputBuilder.buildPipelineModelInput(
        selectedFeatures)

    input_layers_merged = merge(input_layers, mode='concat')

    model = Senna.buildModelGivenInput(
        input_layers_merged, inputs, params, n_out, name_prefix='tac_')

    # SELECT FEATURES WHICH APPEAR IN THE CONFIG
    indices = sorted(
        [selectedFeatures[feature]['pos'] for feature in selectedFeatures])
    model_train = [input_train[i] for i in indices]
    model_dev = [input_dev[i] for i in indices]
    model_test = [input_test[i] for i in indices]

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(events_idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(
        model,
        model_train,
        train_y_cat,
        number_of_epochs,
        params['batch_size'],
        model_dev,
        dev_y,
        model_test,
        test_y,
        measurer=biof1)

    return train_scores, dev_scores, test_scores


def buildAndTrainTempevalModel(learning_params=None, config=[]):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train,
     train_y_cat], [input_dev,
                    dev_y], [input_test,
                             test_y], dicts = TempevalED.readDatasetExt(
                                 params['window_size'], word2Idx, case2Idx)

    [
        _, events_pos2Idx, events_ner2Idx, events_chunking2Idx, events_ace2Idx,
        events_ecb2Idx, events_tac2Idx, _, events_label2Idx, events_idx2Label
    ] = dicts

    [
        events_train_x, events_train_casing_x, events_train_pos_x,
        events_train_ner_x, events_train_chunking_x, events_train_ace_x,
        events_train_ecb_x, events_train_tac_x
    ] = input_train

    n_out = train_y_cat.shape[1]
    n_in_x = events_train_x.shape[1]
    n_in_pos = events_train_pos_x.shape[1]
    n_in_ner = events_train_ner_x.shape[1]
    n_in_chunking = events_train_chunking_x.shape[1]
    n_in_casing = events_train_casing_x.shape[1]
    n_in_ace = events_train_ace_x.shape[1]
    n_in_ecb = events_train_ecb_x.shape[1]
    n_in_tac = events_train_tac_x.shape[1]

    # prepare config
    config_values = {
        'words': {
            'n': n_in_x,
            'idx': embeddings,
            'pos': 0
        },
        'casing': {
            'n': n_in_casing,
            'idx': case2Idx,
            'pos': 1
        },
        'pos': {
            'n': n_in_pos,
            'idx': events_pos2Idx,
            'pos': 2
        },
        'ner': {
            'n': n_in_ner,
            'idx': events_ner2Idx,
            'pos': 3
        },
        'chunking': {
            'n': n_in_chunking,
            'idx': events_chunking2Idx,
            'pos': 4
        },
        'ace': {
            'n': n_in_ace,
            'idx': events_ace2Idx,
            'pos': 5
        },
        'ecb': {
            'n': n_in_ecb,
            'idx': events_ecb2Idx,
            'pos': 6
        },
        'tac': {
            'n': n_in_tac,
            'idx': events_tac2Idx,
            'pos': 7
        }
    }

    selectedFeatures = {key: config_values[key] for key in config}

    input_layers, inputs = InputBuilder.buildPipelineModelInput(
        selectedFeatures)

    input_layers_merged = merge(input_layers, mode='concat')

    model = Senna.buildModelGivenInput(
        input_layers_merged, inputs, params, n_out, name_prefix='tempeval_')

    # SELECT FEATURES WHICH APPEAR IN THE CONFIG
    indices = sorted(
        [selectedFeatures[feature]['pos'] for feature in selectedFeatures])
    model_train = [input_train[i] for i in indices]
    model_dev = [input_dev[i] for i in indices]
    model_test = [input_test[i] for i in indices]

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(events_idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(
        model,
        model_train,
        train_y_cat,
        number_of_epochs,
        params['batch_size'],
        model_dev,
        dev_y,
        model_test,
        test_y,
        measurer=biof1)

    return train_scores, dev_scores, test_scores


def run_models_as_input_exp_with_fixed_params():
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

        if 'ace' in config.tasks:
            #ExperimentHelper.run_build_model('ace', 'pipeline', fixed_params,
            #                buildAndTrainAceModel, 'f1', 'pos',
            #               ['words', 'casing', 'pos'])
            ExperimentHelper.run_build_model(
                'ace', 'pipeline', fixed_params, buildAndTrainAceModel, 'f1',
                'pos', ['words', 'casing', 'pos', 'ner', 'chunking'])
            ExperimentHelper.run_build_model(
                'ace', 'pipeline', fixed_params, buildAndTrainAceModel, 'f1',
                'pos', [
                    'words', 'casing', 'pos', 'ner', 'chunking', 'ecb', 'tac',
                    'tempeval'
                ])
            ExperimentHelper.run_build_model(
                'ace', 'pipeline', fixed_params, buildAndTrainAceModel, 'f1',
                'pos', ['words', 'casing', 'ecb', 'tac', 'tempeval'])
            ExperimentHelper.run_build_model(
                'ace', 'pipeline', fixed_params, buildAndTrainAceModel, 'f1',
                'pos', ['words', 'casing', 'ecb', 'tac'])
            ExperimentHelper.run_build_model(
                'ace', 'pipeline', fixed_params, buildAndTrainAceModel, 'f1',
                'pos', ['words', 'casing', 'ecb', 'tempeval'])
            ExperimentHelper.run_build_model(
                'ace', 'pipeline', fixed_params, buildAndTrainAceModel, 'f1',
                'pos', ['words', 'casing', 'tac', 'tempeval'])
            ExperimentHelper.run_build_model(
                'ace', 'pipeline', fixed_params, buildAndTrainAceModel, 'f1',
                'pos', ['words', 'casing', 'ecb', 'tac'])
            ExperimentHelper.run_build_model('ace', 'pipeline', fixed_params,
                                             buildAndTrainAceModel, 'f1',
                                             'pos', ['words', 'casing', 'ecb'])
            ExperimentHelper.run_build_model('ace', 'pipeline', fixed_params,
                                             buildAndTrainAceModel, 'f1',
                                             'pos', ['words', 'casing', 'tac'])
            ExperimentHelper.run_build_model(
                'ace', 'pipeline', fixed_params, buildAndTrainAceModel, 'f1',
                'pos', ['words', 'casing', 'tempeval'])

        if 'ecb' in config.tasks:
            #ExperimentHelper.run_build_model('ecb', 'pipeline', fixed_params,
            #                buildAndTrainEcbModel, 'f1', 'pos',
            #               ['words', 'casing', 'pos'])
            ExperimentHelper.run_build_model(
                'ecb', 'pipeline', fixed_params, buildAndTrainEcbModel, 'f1',
                'pos', ['words', 'casing', 'pos', 'ner', 'chunking'])
            ExperimentHelper.run_build_model(
                'ecb', 'pipeline', fixed_params, buildAndTrainEcbModel, 'f1',
                'pos', [
                    'words', 'casing', 'pos', 'ner', 'chunking', 'ace', 'tac',
                    'tempeval'
                ])
            ExperimentHelper.run_build_model(
                'ecb', 'pipeline', fixed_params, buildAndTrainEcbModel, 'f1',
                'pos', ['words', 'casing', 'ace', 'tac', 'tempeval'])
            ExperimentHelper.run_build_model(
                'ecb', 'pipeline', fixed_params, buildAndTrainEcbModel, 'f1',
                'pos', ['words', 'casing', 'ace', 'tac'])
            ExperimentHelper.run_build_model(
                'ecb', 'pipeline', fixed_params, buildAndTrainEcbModel, 'f1',
                'pos', ['words', 'casing', 'ace', 'tempeval'])
            ExperimentHelper.run_build_model(
                'ecb', 'pipeline', fixed_params, buildAndTrainEcbModel, 'f1',
                'pos', ['words', 'casing', 'tac', 'tempeval'])
            ExperimentHelper.run_build_model('ecb', 'pipeline', fixed_params,
                                             buildAndTrainEcbModel, 'f1',
                                             'pos', ['words', 'casing', 'ace'])
            ExperimentHelper.run_build_model('ecb', 'pipeline', fixed_params,
                                             buildAndTrainEcbModel, 'f1',
                                             'pos', ['words', 'casing', 'tac'])
            ExperimentHelper.run_build_model(
                'ecb', 'pipeline', fixed_params, buildAndTrainEcbModel, 'f1',
                'pos', ['words', 'casing', 'tempeval'])

        if 'tac' in config.tasks:
            #ExperimentHelper.run_build_model('tac', 'pipeline', fixed_params,
            #                buildAndTrainTacModel, 'f1', 'pos',
            #               ['words', 'casing', 'pos'])
            ExperimentHelper.run_build_model(
                'tac', 'pipeline', fixed_params, buildAndTrainTacModel, 'f1',
                'pos', ['words', 'casing', 'pos', 'ner', 'chunking'])
            ExperimentHelper.run_build_model(
                'tac', 'pipeline', fixed_params, buildAndTrainTacModel, 'f1',
                'pos', [
                    'words', 'casing', 'pos', 'ner', 'chunking', 'ace', 'ecb',
                    'tempeval'
                ])
            ExperimentHelper.run_build_model(
                'tac', 'pipeline', fixed_params, buildAndTrainTacModel, 'f1',
                'pos', ['words', 'casing', 'ace', 'ecb', 'tempeval'])
            ExperimentHelper.run_build_model(
                'tac', 'pipeline', fixed_params, buildAndTrainTacModel, 'f1',
                'pos', ['words', 'casing', 'ace', 'ecb'])
            ExperimentHelper.run_build_model(
                'tac', 'pipeline', fixed_params, buildAndTrainTacModel, 'f1',
                'pos', ['words', 'casing', 'ecb', 'tempeval'])
            ExperimentHelper.run_build_model(
                'tac', 'pipeline', fixed_params, buildAndTrainTacModel, 'f1',
                'pos', ['words', 'casing', 'ace', 'tempeval'])
            ExperimentHelper.run_build_model('tac', 'pipeline', fixed_params,
                                             buildAndTrainTacModel, 'f1',
                                             'pos', ['words', 'casing', 'ecb'])
            ExperimentHelper.run_build_model('tac', 'pipeline', fixed_params,
                                             buildAndTrainTacModel, 'f1',
                                             'pos', ['words', 'casing', 'ace'])
            ExperimentHelper.run_build_model(
                'tac', 'pipeline', fixed_params, buildAndTrainTacModel, 'f1',
                'pos', ['words', 'casing', 'tempeval'])

        if 'tempeval' in config.tasks:
            #ExperimentHelper.run_build_model('tempeval', 'pipeline', fixed_params,
            #                buildAndTrainTempevalModel, 'f1', 'pos',
            #               ['words', 'casing', 'pos'])
            ExperimentHelper.run_build_model(
                'tempeval', 'pipeline', fixed_params,
                buildAndTrainTempevalModel, 'f1', 'pos',
                ['words', 'casing', 'pos', 'ner', 'chunking'])
            ExperimentHelper.run_build_model(
                'tempeval', 'pipeline', fixed_params,
                buildAndTrainTempevalModel, 'f1', 'pos', [
                    'words', 'casing', 'pos', 'ner', 'chunking', 'ace', 'ecb',
                    'tac'
                ])
            ExperimentHelper.run_build_model(
                'tempeval', 'pipeline', fixed_params,
                buildAndTrainTempevalModel, 'f1', 'pos',
                ['words', 'casing', 'ace', 'ecb', 'tac'])
            ExperimentHelper.run_build_model(
                'tempeval', 'pipeline', fixed_params,
                buildAndTrainTempevalModel, 'f1', 'pos',
                ['words', 'casing', 'ecb', 'tac'])
            ExperimentHelper.run_build_model(
                'tempeval', 'pipeline', fixed_params,
                buildAndTrainTempevalModel, 'f1', 'pos',
                ['words', 'casing', 'ace', 'ecb'])
            ExperimentHelper.run_build_model(
                'tempeval', 'pipeline', fixed_params,
                buildAndTrainTempevalModel, 'f1', 'pos',
                ['words', 'casing', 'ace', 'tac'])
            ExperimentHelper.run_build_model('tempeval', 'pipeline',
                                             fixed_params,
                                             buildAndTrainTempevalModel, 'f1',
                                             'pos', ['words', 'casing', 'ecb'])
            ExperimentHelper.run_build_model('tempeval', 'pipeline',
                                             fixed_params,
                                             buildAndTrainTempevalModel, 'f1',
                                             'pos', ['words', 'casing', 'tac'])
            ExperimentHelper.run_build_model('tempeval', 'pipeline',
                                             fixed_params,
                                             buildAndTrainTempevalModel, 'f1',
                                             'pos', ['words', 'casing', 'ace'])


#extendAceED()
#extendEcbED()
#extendTacED()
#extendTempevalED()
run_models_as_input_exp_with_fixed_params()
