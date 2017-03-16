from datasets.conll_ner import CoNLLNer
from datasets.conll_chunking import CoNLLChunking
from datasets.wsj_pos import WSJPos
from datasets.universal_dependencies_pos import UDPos
from embeddings.dependency_based_word_embeddings import DependencyBasedWordEmbeddings as Embeddings
from models import Trainer, InputBuilder
from models.NER import SennaNER as NER
from models.POS import SennaPOS as POS
from models.Chunking import SennaChunking as Chunking
from optimizer import OptimizedModels
from measurements import Measurer
import config
from parameters import parameter_space
from logs import Logger
import random

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

# ----- metric results -----#

# Casing matrix
case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
            'contains_digit': 6, 'PADDING': 7}
n_in_case = len(case2Idx)

# Read in embeddings
embeddings = Embeddings.embeddings
word2Idx = Embeddings.word2Idx


def buildAndTrainNERModelWithPos(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    print 'build pnn ner model with pos'
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = CoNLLNer.readDataset(
        params['window_size'], word2Idx, case2Idx)

    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, caseLookup, label2Idx, idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing,
                                                                params['update_word_embeddings'])

    model_pos = OptimizedModels.getWSJPOSModelGivenInput(input_layers, inputs, window_size=params['window_size'])
    # model_chunking = OptimizedModels.getChunkingModelGivenInput(input_layers, inputs, window_size=params['window_size'])

    model_ner = NER.buildNERModelWithDropoutPNN(input_layers, inputs, params, n_out, additional_models=[model_pos])

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model_ner, input_train,
                                                                                 train_y_cat, number_of_epochs,
                                                                                 params['batch_size'], input_dev,
                                                                                 dev_y, input_test, test_y,
                                                                                 measurements=[biof1])

    return train_scores, dev_scores, test_scores

def buildAndTrainNERModelWithChunking(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    print 'build pnn ner model with chunking'
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = CoNLLNer.readDataset(
        params['window_size'], word2Idx, case2Idx)

    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, caseLookup, label2Idx, idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing,
                                                                params['update_word_embeddings'])

    # model_pos = OptimizedModels.getPOSModelGivenInput(input_layers, inputs, window_size=params['window_size'])
    model_chunking = OptimizedModels.getChunkingModelGivenInput(input_layers, inputs, window_size=params['window_size'])

    model_ner = NER.buildNERModelWithDropoutPNN(input_layers, inputs, params, n_out, additional_models=[model_chunking])

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model_ner, input_train,
                                                                                 train_y_cat, number_of_epochs,
                                                                                 params['batch_size'], input_dev,
                                                                                 dev_y, input_test, test_y,
                                                                                 measurements=[biof1])

    return train_scores, dev_scores, test_scores

def buildAndTrainNERModelWithChunkingPos(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    print 'build pnn ner model with chunking and pos'
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = CoNLLNer.readDataset(
        params['window_size'], word2Idx, case2Idx)

    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, caseLookup, label2Idx, idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing,
                                                                params['update_word_embeddings'])

    model_pos = OptimizedModels.getWSJPOSModelGivenInput(input_layers, inputs, window_size=params['window_size'])
    model_chunking = OptimizedModels.getChunkingModelGivenInput(input_layers, inputs, window_size=params['window_size'])

    model_ner = NER.buildNERModelWithDropoutPNN(input_layers, inputs, params, n_out, additional_models=[model_chunking, model_pos])

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model_ner, input_train,
                                                                                 train_y_cat, number_of_epochs,
                                                                                 params['batch_size'], input_dev,
                                                                                 dev_y, input_test, test_y,
                                                                                 measurements=[biof1])

    return train_scores, dev_scores, test_scores


def buildAndTrainPOSModelWithNer(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params
    print 'build pnn pos model with ner'

    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y] = WSJPos.readDataset(
        params['window_size'], word2Idx, case2Idx)

    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing,
                                                                params['update_word_embeddings'])

    model_ner = OptimizedModels.getNERModelGivenInput(input_layers, inputs, window_size=params['window_size'])
    # model_chunking = OptimizedModels.getChunkingModelGivenInput(input_layers, inputs, window_size=params['window_size'])

    model_pos = POS.buildPOSModelWithDropoutPNN(input_layers, inputs, params, n_out, additional_models=[model_ner])

    # ----- Train Model ----- #
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model_pos, input_train,
                                                                                 train_y_cat, number_of_epochs,
                                                                                 params['batch_size'], input_dev,
                                                                                 dev_y, input_test, test_y,
                                                                                 measurements=[
                                                                                     Measurer.measureAccuracy])

    return train_scores, dev_scores, test_scores

def buildAndTrainPOSModelWithChunking(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    print 'build pnn pos model with chunking'
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y] = WSJPos.readDataset(
        params['window_size'], word2Idx, case2Idx)

    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing,
                                                                params['update_word_embeddings'])

    #model_ner = OptimizedModels.getNERModelGivenInput(input_layers, inputs, window_size=params['window_size'])
    model_chunking = OptimizedModels.getChunkingModelGivenInput(input_layers, inputs, window_size=params['window_size'])

    model_pos = POS.buildPOSModelWithDropoutPNN(input_layers, inputs, params, n_out, additional_models=[model_chunking])

    # ----- Train Model ----- #
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model_pos, input_train,
                                                                                 train_y_cat, number_of_epochs,
                                                                                 params['batch_size'], input_dev,
                                                                                 dev_y, input_test, test_y,
                                                                                 measurements=[
                                                                                     Measurer.measureAccuracy])

    return train_scores, dev_scores, test_scores

def buildAndTrainPOSModelWithChunkingNer(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    print 'build pnn pos model with chunking and ner'
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y] = WSJPos.readDataset(
        params['window_size'], word2Idx, case2Idx)

    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing,
                                                                params['update_word_embeddings'])

    model_ner = OptimizedModels.getNERModelGivenInput(input_layers, inputs, window_size=params['window_size'])
    model_chunking = OptimizedModels.getChunkingModelGivenInput(input_layers, inputs, window_size=params['window_size'])

    model_pos = POS.buildPOSModelWithDropoutPNN(input_layers, inputs, params, n_out, additional_models=[model_ner, model_chunking])

    # ----- Train Model ----- #
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model_pos, input_train,
                                                                                 train_y_cat, number_of_epochs,
                                                                                 params['batch_size'], input_dev,
                                                                                 dev_y, input_test, test_y,
                                                                                 measurements=[
                                                                                     Measurer.measureAccuracy])

    return train_scores, dev_scores, test_scores

def buildAndTrainWSJPOSModelWithUDPos(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    print 'build pnn pos model with chunking and ner'
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y] = WSJPos.readDataset(
        params['window_size'], word2Idx, case2Idx)

    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing,
                                                                params['update_word_embeddings'])

    model_ud_pos = OptimizedModels.getUDPOSModelGivenInput(input_layers, inputs, window_size=params['window_size'])

    model_pos = POS.buildPOSModelWithDropoutPNN(input_layers, inputs, params, n_out, additional_models=[model_ud_pos], name_prefix='wsj_')

    # ----- Train Model ----- #
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model_pos, input_train,
                                                                                 train_y_cat, number_of_epochs,
                                                                                 params['batch_size'], input_dev,
                                                                                 dev_y, input_test, test_y,
                                                                                 measurements=[
                                                                                     Measurer.measureAccuracy])

    return train_scores, dev_scores, test_scores

def buildAndTrainUDPOSModelWithWSJPos(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    print 'build pnn pos model with chunking and ner'
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y] = UDPos.readDataset(
        params['window_size'], word2Idx, case2Idx)

    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing,
                                                                params['update_word_embeddings'])

    model_wsj_pos = OptimizedModels.getWSJPOSModelGivenInput(input_layers, inputs, window_size=params['window_size'])

    model_pos = POS.buildPOSModelWithDropoutPNN(input_layers, inputs, params, n_out, additional_models=[model_wsj_pos], name_prefix='ud_')

    # ----- Train Model ----- #
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model_pos, input_train,
                                                                                 train_y_cat, number_of_epochs,
                                                                                 params['batch_size'], input_dev,
                                                                                 dev_y, input_test, test_y,
                                                                                 measurements=[
                                                                                     Measurer.measureAccuracy])

    return train_scores, dev_scores, test_scores


def buildAndTrainChunkingModelWithPosNer(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    print 'build pnn chunking model with pos and ner'
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = CoNLLChunking.readDataset(
        params['window_size'], word2Idx, case2Idx)

    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, caseLookup, label2Idx, idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing,
                                                                params['update_word_embeddings'])

    model_pos = OptimizedModels.getWSJPOSModelGivenInput(input_layers, inputs, window_size=params['window_size'])
    model_ner = OptimizedModels.getNERModelGivenInput(input_layers, inputs, window_size=params['window_size'])

    model_chunking = Chunking.buildChunkingModelWithDropoutPNN(input_layers, inputs, params, n_out,
                                                        additional_models=[model_pos, model_ner])

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model_chunking, input_train,
                                                                                 train_y_cat, number_of_epochs,
                                                                                 params['batch_size'], input_dev,
                                                                                 dev_y, input_test, test_y,
                                                                                 measurements=[biof1])

    return train_scores, dev_scores, test_scores


def buildAndTrainChunkingModelWithPos(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    print 'build pnn chunking model with pos'
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = CoNLLChunking.readDataset(
        params['window_size'], word2Idx, case2Idx)

    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, caseLookup, label2Idx, idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing,
                                                                params['update_word_embeddings'])

    model_pos = OptimizedModels.getWSJPOSModelGivenInput(input_layers, inputs, window_size=params['window_size'])

    model_chunking = Chunking.buildChunkingModelWithDropoutPNN(input_layers, inputs, params, n_out,
                                                        additional_models=[model_pos])

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model_chunking, input_train,
                                                                                 train_y_cat, number_of_epochs,
                                                                                 params['batch_size'], input_dev,
                                                                                 dev_y, input_test, test_y,
                                                                                 measurements=[biof1])

    return train_scores, dev_scores, test_scores


def buildAndTrainChunkingModelWithNer(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    print 'build pnn chunking model with ner'
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = CoNLLChunking.readDataset(
        params['window_size'], word2Idx, case2Idx)

    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, caseLookup, label2Idx, idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing,
                                                                params['update_word_embeddings'])

    model_ner = OptimizedModels.getNERModelGivenInput(input_layers, inputs, window_size=params['window_size'])

    model_chunking = Chunking.buildChunkingModelWithDropoutPNN(input_layers, inputs, params, n_out,
                                                        additional_models=[model_ner])

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model_chunking, input_train,
                                                                                 train_y_cat, number_of_epochs,
                                                                                 params['batch_size'], input_dev,
                                                                                 dev_y, input_test, test_y,
                                                                                 measurements=[biof1])

    return train_scores, dev_scores, test_scores


def run_pnn_exp_with_random_params():
    max_evals = config.number_of_evals

    for model_nr in xrange(max_evals):
        params = {}
        for key, values in parameter_space.space.iteritems():
            params[key] = random.choice(values)

        print "Model nr. ", model_nr

        print params
        best_train_scores_ner, best_dev_scores_ner, best_test_scores_ner = buildAndTrainNERModelWithPos(params)
        print params
        for (sample_scores, sample) in best_train_scores_ner:
            for score in sample_scores:
                print "Max f1 train ner: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'pnn_adapter', 'ner', 'train', params,
                                                     score[0], score[1], sample, 'pos')
        for (sample_scores, sample) in best_dev_scores_ner:
            for score in sample_scores:
                print "Max f1 dev ner: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'pnn_adapter', 'ner', 'dev', params,
                                                     score[0], score[1], sample, 'pos')
        for (sample_scores, sample) in best_test_scores_ner:
            for score in sample_scores:
                print "Max f1 test ner: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'pnn_adapter', 'ner', 'test', params,
                                                     score[0], score[1], sample, 'pos')

        best_train_scores_pos, best_dev_scores_pos, best_test_scores_pos = buildAndTrainPOSModelWithNer(params)
        print params
        for (sample_scores, sample) in best_train_scores_pos:
            for score in sample_scores:
                print "Max acc train pos: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'pnn_adapter', 'pos', 'train', params,
                                                     score[0], score[1], sample, 'ner')
        for (sample_scores, sample) in best_dev_scores_pos:
            for score in sample_scores:
                print "Max acc dev pos: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'pnn_adapter', 'pos', 'dev', params,
                                                     score[0], score[1], sample, 'ner')
        for (sample_scores, sample) in best_test_scores_pos:
            for score in sample_scores:
                print "Max acc test pos: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'pnn_adapter', 'pos', 'test', params,
                                                     score[0],
                                                     score[1], sample, 'ner')

        best_train_scores_chunking, best_dev_scores_chunking, best_test_scores_chunking = buildAndTrainChunkingModelWithPosNer(
            params)
        print params
        for (sample_scores, sample) in best_train_scores_chunking:
            for score in sample_scores:
                print "Max f1 train chunking: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'pnn_adapter', 'chunking', 'train',
                                                     params,
                                                     score[0], score[1], sample, 'pos-ner')
        for (sample_scores, sample) in best_dev_scores_chunking:
            for score in sample_scores:
                print "Max f1 dev chunking: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'pnn_adapter', 'chunking', 'dev',
                                                     params,
                                                     score[0], score[1], sample, 'pos-ner')
        for (sample_scores, sample) in best_test_scores_chunking:
            for score in sample_scores:
                print "Max f1 test chunking: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'pnn_adapter', 'chunking', 'test',
                                                     params,

                                                     score[0], score[1], sample, 'pos-ner')


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

        if 'ner' in config.tasks:
            run_build_model('ner', 'pnn_dropout', fixed_params, buildAndTrainNERModelWithChunkingPos, 'f1', 'chunking-pos')
            run_build_model('ner', 'pnn_dropout', fixed_params, buildAndTrainNERModelWithChunking, 'f1', 'chunking')
            run_build_model('ner', 'pnn_dropout', fixed_params, buildAndTrainNERModelWithPos, 'f1', 'pos')

        if 'chunking' in config.tasks:
            run_build_model('chunking', 'pnn_dropout', fixed_params, buildAndTrainChunkingModelWithPosNer, 'f1', 'pos-ner')
            run_build_model('chunking', 'pnn_dropout', fixed_params, buildAndTrainChunkingModelWithPos, 'f1', 'pos')
            run_build_model('chunking', 'pnn_dropout', fixed_params, buildAndTrainChunkingModelWithNer, 'f1', 'ner')

        if 'wsj_pos' in config.tasks:
            run_build_model('wsj_pos', 'pnn_dropout', fixed_params, buildAndTrainPOSModelWithChunkingNer, 'acc', 'chunking-ner')
            run_build_model('wsj_pos', 'pnn_dropout', fixed_params, buildAndTrainPOSModelWithChunking, 'acc', 'chunking')
            run_build_model('wsj_pos', 'pnn_dropout', fixed_params, buildAndTrainPOSModelWithNer, 'acc', 'ner')
            run_build_model('wsj_pos', 'pnn_dropout', fixed_params, buildAndTrainWSJPOSModelWithUDPos, 'acc', 'ud_pos')

        if 'ud_pos' in config.tasks:
            run_build_model('ud_pos', 'pnn_dropout', fixed_params, buildAndTrainUDPOSModelWithWSJPos, 'acc', 'wsj_pos')


def run_build_model(task, exp, params, build_model_func, score_name, transfer_models):
    train_scores, dev_scores, test_scores = build_model_func(params)
    print params
    for (sample_scores, sample) in train_scores:
        for score in sample_scores:
            print "Max {0} train {1} with {2}: {3:.4f} in epoch: {4} with samples: {5}".format(score_name, task, transfer_models, score[0], score[1], sample)
            Logger.save_reduced_datasets_results(config.experiments_log_path, exp, task, 'train',
                                                 params, score[0], score[1], sample, transfer_models)
    for (sample_scores, sample) in dev_scores:
        for score in sample_scores:
            print "Max {0} dev {1} with {2}: {3:.4f} in epoch: {4} with samples: {5}".format(score_name, task,
                                                                                          transfer_models, score[0],
                                                                                          score[1], sample)
            Logger.save_reduced_datasets_results(config.experiments_log_path, exp, task, 'dev',
                                                 params, score[0], score[1], sample, transfer_models)
    for (sample_scores, sample) in test_scores:
        for score in sample_scores:
            print "Max {0} test {1} with {2}: {3:.4f} in epoch: {4} with samples: {5}".format(score_name, task,
                                                                                        transfer_models, score[0],
                                                                                        score[1], sample)
            Logger.save_reduced_datasets_results(config.experiments_log_path, exp, task, 'test',
                                                 params, score[0], score[1], sample, transfer_models)

    print '\n\n-------------------- END --------------------\n\n'

run_pnn_exp_with_fixed_params()

