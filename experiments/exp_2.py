from keras.layers import Input, Embedding, Flatten, merge

import embeddings.dependency_based_word_embeddings.DependencyBasedWordEmbeddings as Embeddings
from models import Trainer, InputBuilder
from datasets.conll_ner import CoNLLNer
from datasets.conll_chunking import CoNLLChunking
from datasets.universal_dependencies_pos import UDPos
from models.NER import SennaNER as NER
from models.POS import SennaPOS as POS
from models.Chunking import SennaChunking as Chunking
from optimizer import OptimizedModels
from parameters import parameter_space
from measurements import Measurer
import config
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

best_pos_window_size = 2
best_ner_window_size = 2

number_of_epochs = config.number_of_epochs

# ----- metric results -----#
metric_results = []

#Casing matrix
case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING':7}
n_in_case = len(case2Idx)

# Read in embeddings
embeddings = Embeddings.embeddings
word2Idx = Embeddings.word2Idx

def extendCoNLLNer():
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = CoNLLNer.readDataset(best_pos_window_size, word2Idx, case2Idx)

    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, caseLookup, ner_label2Idx, ner_idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    input_layers_merged, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing)

    pos_model, _, _, _ = OptimizedModels.getPOSModelGivenInput(input_layers_merged, inputs, window_size=best_pos_window_size)

    #pos_model = OptimizedModels.getPOSModel(embeddings, word2Idx)
    pred_train = pos_model.predict(input_train, verbose=0).argmax(axis=-1)
    pred_dev = pos_model.predict(input_dev, verbose=0).argmax(axis=-1)
    pred_test = pos_model.predict(input_test, verbose=0).argmax(axis=-1)

    pos_label2Idx, pos_idx2Label = UDPos.getLabelDict()
    pred_train_labels = map(lambda idx: pos_idx2Label[idx], pred_train)
    pred_dev_labels = map(lambda idx: pos_idx2Label[idx], pred_dev)
    pred_test_labels = map(lambda idx: pos_idx2Label[idx], pred_test)

    train_extensions = [pred_train_labels]
    dev_extensions = [pred_dev_labels]
    test_extensions = [pred_test_labels]

    CoNLLNer.extendDataset("./datasets/conll_ner/data/eng.conllu", train_extensions, dev_extensions, test_extensions)

def extendUDPOS():
    # Read in files
    word2Idx = Embeddings.word2Idx
    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y] = UDPos.readDataset(best_ner_window_size, word2Idx, case2Idx)

    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    input_layers_merged, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing)

    ner_model, _, _, _ = OptimizedModels.getNERModelGivenInput(input_layers_merged, inputs,
                                                            window_size=best_ner_window_size)
    pred_train = ner_model.predict(input_train, verbose=0).argmax(axis=-1)
    pred_dev = ner_model.predict(input_dev, verbose=0).argmax(axis=-1)
    pred_test = ner_model.predict(input_test, verbose=0).argmax(axis=-1)

    ner_label2Idx, ner_idx2Label = CoNLLNer.getLabelDict()
    pred_train_labels = map(lambda idx: ner_idx2Label[idx], pred_train)
    pred_dev_labels = map(lambda idx: ner_idx2Label[idx], pred_dev)
    pred_test_labels = map(lambda idx: ner_idx2Label[idx], pred_test)

    train_extensions = [pred_train_labels]
    dev_extensions = [pred_dev_labels]
    test_extensions = [pred_test_labels]

    UDPos.extendDataset("./datasets/universal_dependencies_pos/data/en-ud.conllu", train_extensions, dev_extensions, test_extensions)

def extendCoNLLChunking():
    # ----- read Data for pos with best pos window ----- #
    [input_train_for_pos, train_y_cat_for_pos], [input_dev_for_pos, dev_y_for_pos], [input_test_for_pos, test_y_for_pos], dicts_for_pos = CoNLLChunking.readDataset(best_pos_window_size, word2Idx, case2Idx)

    # calculate dims for model building
    [train_x_for_pos, train_case_x_for_pos] = input_train_for_pos
    n_in_x_for_pos = train_x_for_pos.shape[1]
    n_in_casing_for_pos = train_case_x_for_pos.shape[1]

    # build pos model
    input_layers_for_pos, inputs_for_pos = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x_for_pos, n_in_casing_for_pos)
    pos_model, _, _, _ = OptimizedModels.getPOSModelGivenInput(input_layers_for_pos, inputs_for_pos, window_size=best_pos_window_size)

    # predict pos on chunking data
    pos_pred_train = pos_model.predict(input_train_for_pos, verbose=0).argmax(axis=-1)
    pos_pred_dev = pos_model.predict(input_dev_for_pos, verbose=0).argmax(axis=-1)
    pos_pred_test = pos_model.predict(input_test_for_pos, verbose=0).argmax(axis=-1)

    #
    pos_label2Idx, pos_idx2Label = UDPos.getLabelDict()
    pos_pred_train_labels = map(lambda idx: pos_idx2Label[idx], pos_pred_train)
    pos_pred_dev_labels = map(lambda idx: pos_idx2Label[idx], pos_pred_dev)
    pos_pred_test_labels = map(lambda idx: pos_idx2Label[idx], pos_pred_test)

    # ----- read Data for ner with best ner window ----- #
    [input_train_for_ner, train_y_cat_for_ner], [input_dev_for_ner, dev_y_for_ner], [input_test_for_ner, test_y_for_ner], dicts_for_ner = CoNLLChunking.readDataset(best_ner_window_size, word2Idx, case2Idx)

    # calculate dims for model building
    [train_x_for_ner, train_case_x_for_ner] = input_train_for_ner
    n_in_x_for_ner = train_x_for_ner.shape[1]
    n_in_casing_for_ner = train_case_x_for_ner.shape[1]

    # build pos model
    input_layers_for_ner, inputs_for_ner = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x_for_ner,
                                                                                n_in_casing_for_ner)
    ner_model, _, _, _ = OptimizedModels.getNERModelGivenInput(input_layers_for_ner, inputs_for_ner, window_size=best_ner_window_size)

    # predict ner on chunking data
    ner_pred_train = ner_model.predict(input_train_for_ner, verbose=0).argmax(axis=-1)
    ner_pred_dev = ner_model.predict(input_dev_for_ner, verbose=0).argmax(axis=-1)
    ner_pred_test = ner_model.predict(input_test_for_ner, verbose=0).argmax(axis=-1)

    #
    ner_label2Idx, ner_idx2Label = CoNLLNer.getLabelDict()
    ner_pred_train_labels = map(lambda idx: ner_idx2Label[idx], ner_pred_train)
    ner_pred_dev_labels = map(lambda idx: ner_idx2Label[idx], ner_pred_dev)
    ner_pred_test_labels = map(lambda idx: ner_idx2Label[idx], ner_pred_test)

    train_extensions = [pos_pred_train_labels, ner_pred_train_labels]
    dev_extensions = [pos_pred_dev_labels, ner_pred_dev_labels]
    test_extensions = [pos_pred_test_labels, ner_pred_test_labels]

    CoNLLChunking.extendDataset("./datasets/conll_chunking/data/chunking.conllu", train_extensions, dev_extensions, test_extensions)

def buildAndTrainNERModel(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train, ner_train_y_cat], [input_dev, ner_dev_y], [input_test, ner_test_y], dicts = CoNLLNer.readDatasetExt(params['window_size'], word2Idx, case2Idx)

    [ner_train_x, ner_train_pos_x, ner_train_casing_x] = input_train
    [ner_dev_x, ner_dev_pos_x, ner_dev_casing_x] = input_dev
    [ner_test_x, ner_test_pos_x, ner_test_casing_x] = input_test
    [_, pos2Idx, _, ner_label2Idx, ner_idx2Label] = dicts


    ner_n_out = ner_train_y_cat.shape[1]
    n_in_x = ner_train_x.shape[1]
    n_in_pos = ner_train_pos_x.shape[1]
    n_in_casing = ner_train_casing_x.shape[1]


    words_input = Input(shape=(n_in_x,), dtype='int32', name='words_input')
    wordEmbeddingLayer = Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in_x,
                                   weights=[embeddings], trainable=False)
    words = wordEmbeddingLayer(words_input)
    words = Flatten(name='words_flatten')(words)

    pos_input = Input(shape=(n_in_pos,), dtype='int32', name='pos_input')
    posEmbeddingLayer = Embedding(output_dim=len(pos2Idx), input_dim=len(pos2Idx), input_length=n_in_pos,
                                   trainable=True)
    pos = posEmbeddingLayer(pos_input)
    pos = Flatten(name='pos_flatten')(pos)

    case_input = Input(shape=(n_in_x,), dtype='int32', name='case_input')
    caseEmbeddingLayer = Embedding(output_dim=len(case2Idx), input_dim=len(case2Idx), input_length=n_in_casing,
                                   trainable=True)
    casing = caseEmbeddingLayer(case_input)
    casing = Flatten(name='casing_flatten')(casing)

    input_layers = [words, pos, casing]
    inputs = [words_input, pos_input, case_input]

    input_layers_merged = merge(input_layers, mode='concat')

    model = NER.buildNERModelGivenInput(input_layers_merged, inputs, params, ner_n_out)

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(ner_idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, input_train,
                                                                   ner_train_y_cat, number_of_epochs,
                                                                   params['batch_size'], input_dev,
                                                                   ner_dev_y, input_test, ner_test_y,
                                                                   measurements=[biof1])

    return train_scores, dev_scores, test_scores

def buildAndTrainChunkingModel(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = CoNLLChunking.readDatasetExt(params['window_size'], word2Idx, case2Idx)

    [chunking_train_x, chunking_train_pos_x, chunking_train_ner_x, chunking_train_casing_x] = input_train
    [chunking_dev_x, chunking_dev_pos_x, chunking_dev_ner_x, chunking_dev_casing_x] = input_dev
    [chunking_test_x, chunking_test_pos_x, chunking_test_ner_x, chunking_test_casing_x] = input_test
    [_, pos2Idx, ner2Idx, _, chunking_label2Idx, chunking_idx2Label] = dicts

    chunking_n_out = train_y_cat.shape[1]
    n_in_x = chunking_train_x.shape[1]
    n_in_pos = chunking_train_pos_x.shape[1]
    n_in_ner = chunking_train_ner_x.shape[1]
    n_in_casing = chunking_train_casing_x.shape[1]

    words_input = Input(shape=(n_in_x,), dtype='int32', name='words_input')
    wordEmbeddingLayer = Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in_x,
                                   weights=[embeddings], trainable=False)
    words = wordEmbeddingLayer(words_input)
    words = Flatten(name='words_flatten')(words)

    pos_input = Input(shape=(n_in_pos,), dtype='int32', name='pos_input')
    posEmbeddingLayer = Embedding(output_dim=len(pos2Idx), input_dim=len(pos2Idx), input_length=n_in_pos,
                                   trainable=True)
    pos = posEmbeddingLayer(pos_input)
    pos = Flatten(name='pos_flatten')(pos)

    ner_input = Input(shape=(n_in_ner,), dtype='int32', name='ner_input')
    nerEmbeddingLayer = Embedding(output_dim=len(ner2Idx), input_dim=len(ner2Idx), input_length=n_in_ner,
                                   trainable=True)
    ner = nerEmbeddingLayer(ner_input)
    ner = Flatten(name='ner_flatten')(ner)

    case_input = Input(shape=(n_in_x,), dtype='int32', name='case_input')
    caseEmbeddingLayer = Embedding(output_dim=len(case2Idx), input_dim=len(case2Idx), input_length=n_in_casing,
                                   trainable=True)
    casing = caseEmbeddingLayer(case_input)
    casing = Flatten(name='casing_flatten')(casing)

    input_layers = [words, pos, ner, casing]
    inputs = [words_input, pos_input, ner_input, case_input]

    input_layers_merged = merge(input_layers, mode='concat')

    model = Chunking.buildChunkingModelGivenInput(input_layers_merged, inputs, params, chunking_n_out)

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(chunking_idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, input_train,
                                                                   train_y_cat, number_of_epochs,
                                                                   params['batch_size'], input_dev,
                                                                   dev_y, input_test, test_y,
                                                                   measurements=[biof1])

    return train_scores, dev_scores, test_scores


def buildAndTrainPOSModel(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = UDPos.readDatasetExt(params['window_size'], word2Idx, case2Idx)

    [pos_train_x, pos_train_ner_x, pos_train_casing_x] = input_train
    [pos_dev_x, pos_dev_ner_x, pos_dev_casing_x] = input_dev
    [pos_test_x, pos_test_ner_x, pos_test_casing_x] = input_test
    [_, ner2Idx, _, pos_label2Idx, pos_idx2Label] = dicts

    pos_n_out = train_y_cat.shape[1]
    n_in_x = pos_train_x.shape[1]
    n_in_ner = pos_train_ner_x.shape[1]
    n_in_casing = pos_train_casing_x.shape[1]


    words_input = Input(shape=(n_in_x,), dtype='int32', name='words_input')
    wordEmbeddingLayer = Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in_x,
                                   weights=[embeddings], trainable=False)
    words = wordEmbeddingLayer(words_input)
    words = Flatten(name='words_flatten')(words)

    ner_input = Input(shape=(n_in_ner,), dtype='int32', name='ner_input')
    nerEmbeddingLayer = Embedding(output_dim=len(ner2Idx), input_dim=len(ner2Idx), input_length=n_in_ner,
                                   trainable=True)
    ner = nerEmbeddingLayer(ner_input)
    ner = Flatten(name='ner_flatten')(ner)

    case_input = Input(shape=(n_in_x,), dtype='int32', name='case_input')
    caseEmbeddingLayer = Embedding(output_dim=len(case2Idx), input_dim=len(case2Idx), input_length=n_in_casing,
                                   trainable=True)
    casing = caseEmbeddingLayer(case_input)
    casing = Flatten(name='casing_flatten')(casing)

    input_layers = [words, ner, casing]
    inputs = [words_input, ner_input, case_input]

    input_layers_merged = merge(input_layers, mode='concat')

    model = POS.buildPosModelGivenInput(input_layers_merged, inputs, params, pos_n_out)

    # ----- Train Model ----- #
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, input_train,
                                                                   train_y_cat, number_of_epochs,
                                                                   params['batch_size'], input_dev,
                                                                   dev_y, input_test, test_y,
                                                                   measurements=[Measurer.measureAccuracy])

    return train_scores, dev_scores, test_scores


def run_models_as_input_exp_with_random_params():
    max_evals = config.number_of_evals

    for model_nr in xrange(max_evals):
        params = {}
        for key, values in parameter_space.space.iteritems():
            params[key] = random.choice(values)

        print "Model nr. ", model_nr
        print params
        best_train_scores_ner, best_dev_scores_ner, best_test_scores_ner = buildAndTrainNERModel(params)
        print params
        for (sample_scores, sample) in best_train_scores_ner:
            for score in sample_scores:
                print "Max f1 train ner: %.4f in epoch: %d with samples: %d" % (score[0][2], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'ner', 'train', params, score[0][2], score[1], sample, 'pos')
        for (sample_scores, sample) in best_dev_scores_ner:
            for score in sample_scores:
                print "Max f1 dev ner: %.4f in epoch: %d with samples: %d" % (score[0][2], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'ner', 'dev', params, score[0][2], score[1], sample, 'pos')
        for (sample_scores, sample) in best_test_scores_ner:
            for score in sample_scores:
                print "Max f1 test ner: %.4f in epoch: %d with samples: %d" % (score[0][2], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'ner', 'test', params, score[0][2], score[1], sample, 'pos')

        best_train_scores_chunking, best_dev_scores_chunking, best_test_scores_chunking = buildAndTrainChunkingModel(params)
        print params
        for (sample_scores, sample) in best_train_scores_chunking:
            for score in sample_scores:
                print "Max f1 train chunking: %.4f in epoch: %d with samples: %d" % (score[0][2], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'chunking', 'train', params,
                                                     score[0][2], score[1], sample, 'pos-ner')
        for (sample_scores, sample) in best_dev_scores_chunking:
            for score in sample_scores:
                print "Max f1 dev chunking: %.4f in epoch: %d with samples: %d" % (score[0][2], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'chunking', 'dev', params,
                                                     score[0][2], score[1], sample, 'pos-ner')
        for (sample_scores, sample) in best_test_scores_chunking:
            for score in sample_scores:
                print "Max f1 test chunking: %.4f in epoch: %d with samples: %d" % (score[0][2], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'chunking', 'test', params,
                                                     score[0][2], score[1], sample, 'pos-ner')

        best_train_scores_pos, best_dev_scores_pos, best_test_scores_pos = buildAndTrainPOSModel(params)
        print params
        for (sample_scores, sample) in best_train_scores_pos:
            for score in sample_scores:
                print "Max acc train pos: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'pos', 'train', params, score[0], score[1], sample, 'ner')
        for (sample_scores, sample) in best_dev_scores_pos:
            for score in sample_scores:
                print "Max acc dev pos: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'pos', 'dev', params, score[0], score[1], sample, 'ner')
        for (sample_scores, sample) in best_test_scores_pos:
            for score in sample_scores:
                print "Max acc test pos: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'pos', 'test', params, score[0],
                                                     score[1], sample, 'ner')
def run_models_as_input_exp_with_fixed_params():
    fixed_params = {
        'update_word_embeddings': False,
        'window_size': 3,
        'batch_size': 128,
        'hidden_dims': 100,
        'activation': 'relu',
        'dropout': 0.3,
        'optimizer': 'adam',
        'number_of_epochs': [config.number_of_epochs]
    }

    best_train_scores_ner, best_dev_scores_ner, best_test_scores_ner = buildAndTrainNERModel(fixed_params)
    print fixed_params
    for (sample_scores, sample) in best_train_scores_ner:
        for score in sample_scores:
            print "Max f1 train ner: %.4f in epoch: %d with samples: %d" % (score[0][2], sample, score[1])
            Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'ner', 'train', fixed_params, score[0][2], score[1], sample, 'pos')
    for (sample_scores, sample) in best_dev_scores_ner:
        for score in sample_scores:
            print "Max f1 dev ner: %.4f in epoch: %d with samples: %d" % (score[0][2], sample, score[1])
            Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'ner', 'dev', fixed_params, score[0][2], score[1], sample, 'pos')
    for (sample_scores, sample) in best_test_scores_ner:
        for score in sample_scores:
            print "Max f1 test ner: %.4f in epoch: %d with samples: %d" % (score[0][2], sample, score[1])
            Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'ner', 'test', fixed_params, score[0][2], score[1], sample, 'pos')

    best_train_scores_chunking, best_dev_scores_chunking, best_test_scores_chunking = buildAndTrainChunkingModel(fixed_params)
    print fixed_params
    for (sample_scores, sample) in best_train_scores_chunking:
        for score in sample_scores:
            print "Max f1 train chunking: %.4f in epoch: %d with samples: %d" % (score[0][2], sample, score[1])
            Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'chunking', 'train', fixed_params,
                                                 score[0][2], score[1], sample, 'pos-ner')
    for (sample_scores, sample) in best_dev_scores_chunking:
        for score in sample_scores:
            print "Max f1 dev chunking: %.4f in epoch: %d with samples: %d" % (score[0][2], sample, score[1])
            Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'chunking', 'dev', fixed_params,
                                                 score[0][2], score[1], sample, 'pos-ner')
    for (sample_scores, sample) in best_test_scores_chunking:
        for score in sample_scores:
            print "Max f1 test chunking: %.4f in epoch: %d with samples: %d" % (score[0][2], sample, score[1])
            Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'chunking', 'test', fixed_params,
                                                 score[0][2], score[1], sample, 'pos-ner')

    best_train_scores_pos, best_dev_scores_pos, best_test_scores_pos = buildAndTrainPOSModel(fixed_params)
    print fixed_params
    for (sample_scores, sample) in best_train_scores_pos:
        for score in sample_scores:
            print "Max acc train pos: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
            Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'pos', 'train', fixed_params, score[0], score[1], sample, 'ner')
    for (sample_scores, sample) in best_dev_scores_pos:
        for score in sample_scores:
            print "Max acc dev pos: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
            Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'pos', 'dev', fixed_params, score[0], score[1], sample, 'ner')
    for (sample_scores, sample) in best_test_scores_pos:
        for score in sample_scores:
            print "Max acc test pos: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
            Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'pos', 'test', fixed_params, score[0],
                                                 score[1], sample, 'ner')

#run_models_as_input_exp_with_random_params()
#run_models_as_input_exp_with_fixed_params()
extendCoNLLNer()
extendUDPOS()
extendCoNLLChunking()
