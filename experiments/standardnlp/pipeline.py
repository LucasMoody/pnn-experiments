from keras.layers import Input, Embedding, Flatten, merge

import embeddings.dependency_based_word_embeddings.DependencyBasedWordEmbeddings as Embeddings
from models import Trainer, InputBuilder
from datasets.wsj_pos import WSJPos
from datasets.universal_dependencies_pos import UDPos
from datasets.conll_ner import CoNLLNer
from datasets.conll_chunking import CoNLLChunking
from models import Senna
from optimizer import OptimizedModels
from parameters import parameter_space
from measurements import Measurer
import config
from experiments import ExperimentHelper

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


def extendWSJPos():
    extendED(['pos', 'ner', 'chunking', 'ud_pos'], WSJPos)


def extendUDPos():
    extendED(['ner', 'chunking', 'pos'], UDPos)


def extendNER():
    extendED(['pos', 'chunking'], CoNLLNer)


def extendChunking():
    extendED(['pos', 'ner'], CoNLLChunking)


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

        elif 'ud_pos' == ds:
            # ----- labels from pos  ----- #
            pred_train_labels, pred_dev_labels, pred_test_labels = extendHelper(
                dataset.readDataset, word2Idx, case2Idx,
                best_pos_window_size, OptimizedModels.getUDPOSModelGivenInput,
                UDPos.getLabelDict())
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

    dataset.extendDataset(train_extensions, dev_extensions, test_extensions)

    print '\n--------------------\n          DONE\n--------------------\n'


def buildAndTrainWSJPosModel(learning_params, config=[]):
    params = learning_params

    [input_train,
     train_y_cat], [input_dev,
                    dev_y], [input_test, test_y], dicts = WSJPos.readDatasetExt(
                        params['window_size'], word2Idx, case2Idx)

    [_, _, pos_ner2Idx, pos_chunking2Idx, pos_ud_pos2Idx, pos_label2Idx, pos_idx2Label] = dicts

    [pos_train_x, pos_train_casing_x, pos_train_ner_x, pos_train_chunking_x, pos_train_ud_pos_x] = input_train

    n_out = train_y_cat.shape[1]
    n_in_x = pos_train_x.shape[1]
    n_in_ner = pos_train_ner_x.shape[1]
    n_in_chunking = pos_train_chunking_x.shape[1]
    n_in_ud_pos = pos_train_ud_pos_x.shape[1]
    n_in_casing = pos_train_casing_x.shape[1]

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
        'ner': {
            'n': n_in_ner,
            'idx': pos_ner2Idx,
            'pos': 2
        },
        'chunking': {
            'n': n_in_chunking,
            'idx': pos_chunking2Idx,
            'pos': 3
        },
        'ud_pos': {
            'n': n_in_ud_pos,
            'idx': pos_ud_pos2Idx,
            'pos': 4
        }
    }

    selectedFeatures = {key: config_values[key] for key in config}

    input_layers, inputs = InputBuilder.buildPipelineModelInput(
        selectedFeatures)

    input_layers_merged = merge(input_layers, mode='concat')

    model = Senna.buildModelGivenInput(
        input_layers_merged, inputs, params, n_out, name_prefix='wsj_pos_')

    # SELECT FEATURES WHICH APPEAR IN THE CONFIG
    indices = sorted(
        [selectedFeatures[feature]['pos'] for feature in selectedFeatures])
    model_train = [input_train[i] for i in indices]
    model_dev = [input_dev[i] for i in indices]
    model_test = [input_test[i] for i in indices]

    # ----- Train Model ----- #
    return Trainer.trainModelWithIncreasingData(
        model,
        model_train,
        train_y_cat,
        number_of_epochs,
        params['batch_size'],
        model_dev,
        dev_y,
        model_test,
        test_y,
        measurer=Measurer.measureAccuracy)

def buildAndTrainUDPosModel(learning_params, config=[]):
    params = learning_params

    [input_train,
     train_y_cat], [input_dev,
                    dev_y], [input_test,
                             test_y], dicts = UDPos.readDatasetExt(
                                 params['window_size'], word2Idx, case2Idx)

    [_, _, pos_ner2Idx, pos_chunking2Idx, pos_wsj_pos2Idx, pos_label2Idx, pos_idx2Label] = dicts

    [pos_train_x, pos_train_casing_x, pos_train_ner_x, pos_train_chunking_x, pos_train_wsj_pos_x] = input_train

    n_out = train_y_cat.shape[1]
    n_in_x = pos_train_x.shape[1]
    n_in_ner = pos_train_ner_x.shape[1]
    n_in_chunking = pos_train_chunking_x.shape[1]
    n_in_wsj_pos = pos_train_wsj_pos_x.shape[1]
    n_in_casing = pos_train_casing_x.shape[1]

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
        'ner': {
            'n': n_in_ner,
            'idx': pos_ner2Idx,
            'pos': 2
        },
        'chunking': {
            'n': n_in_chunking,
            'idx': pos_chunking2Idx,
            'pos': 3
        },
        'pos': {
            'n': n_in_wsj_pos,
            'idx': pos_wsj_pos2Idx,
            'pos': 4
        }
    }

    selectedFeatures = {key: config_values[key] for key in config}

    input_layers, inputs = InputBuilder.buildPipelineModelInput(
        selectedFeatures)

    input_layers_merged = merge(input_layers, mode='concat')

    model = Senna.buildModelGivenInput(
        input_layers_merged, inputs, params, n_out, name_prefix='ud_pos_')

    # SELECT FEATURES WHICH APPEAR IN THE CONFIG
    indices = sorted(
        [selectedFeatures[feature]['pos'] for feature in selectedFeatures])
    model_train = [input_train[i] for i in indices]
    model_dev = [input_dev[i] for i in indices]
    model_test = [input_test[i] for i in indices]

    # ----- Train Model ----- #
    return Trainer.trainModelWithIncreasingData(
        model,
        model_train,
        train_y_cat,
        number_of_epochs,
        params['batch_size'],
        model_dev,
        dev_y,
        model_test,
        test_y,
        measurer=Measurer.measureAccuracy)

def buildAndTrainNerModel(learning_params, config=[]):
    params = learning_params

    [input_train,
     train_y_cat], [input_dev,
                    dev_y], [input_test, test_y], dicts = CoNLLNer.readDatasetExt(
                        params['window_size'], word2Idx, case2Idx)

    [_, _, ner_pos2Idx, ner_chunking2Idx, ner_label2Idx, ner_idx2Label] = dicts

    [ner_train_x, ner_train_casing_x, ner_train_pos_x, ner_train_chunking_x] = input_train

    n_out = train_y_cat.shape[1]
    n_in_x = ner_train_x.shape[1]
    n_in_pos = ner_train_pos_x.shape[1]
    n_in_chunking = ner_train_chunking_x.shape[1]
    n_in_casing = ner_train_casing_x.shape[1]

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
            'idx': ner_pos2Idx,
            'pos': 2
        },
        'chunking': {
            'n': n_in_chunking,
            'idx': ner_chunking2Idx,
            'pos': 3
        }
    }

    selectedFeatures = {key: config_values[key] for key in config}

    input_layers, inputs = InputBuilder.buildPipelineModelInput(
        selectedFeatures)

    input_layers_merged = merge(input_layers, mode='concat')

    model = Senna.buildModelGivenInput(
        input_layers_merged, inputs, params, n_out, name_prefix='ner_')

    # SELECT FEATURES WHICH APPEAR IN THE CONFIG
    indices = sorted(
        [selectedFeatures[feature]['pos'] for feature in selectedFeatures])
    model_train = [input_train[i] for i in indices]
    model_dev = [input_dev[i] for i in indices]
    model_test = [input_test[i] for i in indices]

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(ner_idx2Label)
    return Trainer.trainModelWithIncreasingData(
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

def buildAndTrainChunkingModel(learning_params, config=[]):
    params = learning_params

    [input_train,
     train_y_cat], [input_dev,
                    dev_y], [input_test,
                             test_y], dicts = CoNLLChunking.readDatasetExt(
                                 params['window_size'], word2Idx, case2Idx)

    [_, _, chunking_pos2Idx, chunking_ner2Idx, chunking_label2Idx, chunking_idx2Label] = dicts

    [chunking_train_x, chunking_train_casing_x, chunking_train_pos_x, chunking_train_ner_x] = input_train

    n_out = train_y_cat.shape[1]
    n_in_x = chunking_train_x.shape[1]
    n_in_pos = chunking_train_pos_x.shape[1]
    n_in_ner = chunking_train_ner_x.shape[1]
    n_in_casing = chunking_train_casing_x.shape[1]

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
            'idx': chunking_pos2Idx,
            'pos': 2
        },
        'ner': {
            'n': n_in_ner,
            'idx': chunking_ner2Idx,
            'pos': 3
        }
    }

    selectedFeatures = {key: config_values[key] for key in config}

    input_layers, inputs = InputBuilder.buildPipelineModelInput(
        selectedFeatures)

    input_layers_merged = merge(input_layers, mode='concat')

    model = Senna.buildModelGivenInput(
        input_layers_merged, inputs, params, n_out, name_prefix='chunking_')

    # SELECT FEATURES WHICH APPEAR IN THE CONFIG
    indices = sorted(
        [selectedFeatures[feature]['pos'] for feature in selectedFeatures])
    model_train = [input_train[i] for i in indices]
    model_dev = [input_dev[i] for i in indices]
    model_test = [input_test[i] for i in indices]

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(chunking_idx2Label)
    return Trainer.trainModelWithIncreasingData(
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
            runChunkingExp(fixed_params, ['pos', 'ner'])
            runChunkingExp(fixed_params, ['pos'])
            runChunkingExp(fixed_params, ['ner'])

        if 'ud_pos' in config.tasks:
            runUDPosExp(fixed_params, ['pos'])


def runWSJPosExp(params, config):
    ExperimentHelper.run_build_model(
        'wsj_pos',
        'pipeline',
        params,
        buildAndTrainWSJPosModel,
        'acc',
        transfer_config=['words', 'casing'] + config)


def runUDPosExp(params, config):
    ExperimentHelper.run_build_model(
        'ud_pos',
        'pipeline',
        params,
        buildAndTrainUDPosModel,
        'acc',
        transfer_config=['words', 'casing'] + config)


def runNerExp(params, config):
    ExperimentHelper.run_build_model(
        'ner',
        'pipeline',
        params,
        buildAndTrainNerModel,
        'f1',
        transfer_config=['words', 'casing'] + config)


def runChunkingExp(params, config):
    ExperimentHelper.run_build_model(
        'chunking',
        'pipeline',
        params,
        buildAndTrainChunkingModel,
        'f1',
        transfer_config=['words', 'casing'] + config)


#extendWSJPos()
#extendUDPos()
#extendNER()
#extendChunking()
run_models_as_input_exp_with_fixed_params()
