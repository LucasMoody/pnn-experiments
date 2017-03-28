from keras.layers import Input, Embedding, Flatten, merge

import embeddings.dependency_based_word_embeddings.DependencyBasedWordEmbeddings as Embeddings
from models import Trainer, InputBuilder
from datasets.ace_ed import ACEED
from datasets.tac2015_ed import TACED
from models import Senna
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

best_tac_window_size = 3
best_tempeval_window_size = 3
best_ace_window_size = 3
best_ecb_window_size = 3

number_of_epochs = config.number_of_epochs

# ----- metric results -----#
metric_results = []

#Casing matrix
case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING':7}
n_in_case = len(case2Idx)

# Read in embeddings
embeddings = Embeddings.embeddings
word2Idx = Embeddings.word2Idx

def extendHelper(reader, word2Idx, case2Idx, best_window_size, pipeline_model_builder, pipeline_dict):
    # ----- read Data with best window ----- #
    [input_train, events_train_y_cat], [input_dev, events_dev_y], [input_test, events_test_y], dicts = reader(best_window_size, word2Idx, case2Idx)
    # calculate dims for model building
    [train_x, train_case_x] = input_train
    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    # build pos model
    input_layers, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing)
    model_pipeline = pipeline_model_builder(input_layers, inputs, window_size=best_window_size)

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
    # ----- labels from tac for ace ----- #
    pred_train_labels_for_tac, pred_dev_labels_for_tac, pred_test_labels_for_tac = extendHelper(ACEED.readDataset, word2Idx, case2Idx, best_tac_window_size, OptimizedModels., TACED.getLabelDict())

    train_extensions = [pred_train_labels_for_tac]
    dev_extensions = [pred_dev_labels_for_tac]
    test_extensions = [pred_test_labels_for_tac]

    ACEED.extendDataset("./datasets/ace_ed/data/events.conllu", train_extensions, dev_extensions, test_extensions)

def buildAndTrainAceModelWithEcbTacTempeval(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train, ner_train_y_cat], [input_dev, ner_dev_y], [input_test, ner_test_y], dicts = CoNLLNer.readDatasetExt(params['window_size'], word2Idx, case2Idx)

    [ner_train_x, ner_train_pos_x, ner_train_chunking_x, ner_train_casing_x] = input_train
    [ner_dev_x, ner_dev_pos_x, ner_dev_chunking_x, ner_dev_casing_x] = input_dev
    [ner_test_x, ner_test_pos_x, ner_test_chunking_x, ner_test_casing_x] = input_test
    [_, pos2Idx, chunking2Idx, _, ner_label2Idx, ner_idx2Label] = dicts

    ner_n_out = ner_train_y_cat.shape[1]
    n_in_x = ner_train_x.shape[1]
    n_in_pos = ner_train_pos_x.shape[1]
    n_in_chunking = ner_train_chunking_x.shape[1]
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

    chunking_input = Input(shape=(n_in_chunking,), dtype='int32', name='chunking_input')
    chunkingEmbeddingLayer = Embedding(output_dim=len(chunking2Idx), input_dim=len(chunking2Idx), input_length=n_in_chunking,
                                   trainable=True)
    chunking = chunkingEmbeddingLayer(chunking_input)
    chunking = Flatten(name='chunking_flatten')(chunking)

    case_input = Input(shape=(n_in_x,), dtype='int32', name='case_input')
    caseEmbeddingLayer = Embedding(output_dim=len(case2Idx), input_dim=len(case2Idx), input_length=n_in_casing,
                                   trainable=True)
    casing = caseEmbeddingLayer(case_input)
    casing = Flatten(name='casing_flatten')(casing)

    input_layers = [words, pos, chunking, casing]
    inputs = [words_input, pos_input, chunking_input, case_input]

    input_layers_merged = merge(input_layers, mode='concat')

    model = Senna.buildModelGivenInput(input_layers_merged, inputs, params, ner_n_out, name_prefix='ner_')

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(ner_idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, input_train,
                                                                   ner_train_y_cat, number_of_epochs,
                                                                   params['batch_size'], input_dev,
                                                                   ner_dev_y, input_test, ner_test_y,
                                                                   measurements=[biof1])

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

        if 'ner' in config.tasks:
            run_build_model('ner', 'pipeline', fixed_params, buildAndTrainNERModelWithPos, 'f1', 'pos')
            run_build_model('ner', 'pipeline', fixed_params, buildAndTrainNERModelWithChunking, 'f1', 'chunking')
            run_build_model('ner', 'pipeline', fixed_params, buildAndTrainNERModelWithChunkingPos, 'f1', 'chunking-pos')
        if 'wsj_pos' in config.tasks:
            run_build_model('wsj_pos', 'pipeline', fixed_params, buildAndTrainWSJPosModelWithNer, 'acc', 'ner')
            run_build_model('wsj_pos', 'pipeline', fixed_params, buildAndTrainWSJPosModelWithChunking, 'acc', 'chunking')
            run_build_model('wsj_pos', 'pipeline', fixed_params, buildAndTrainWSJPosModelWithChunkingNer, 'acc', 'chunking-ner')
            run_build_model('wsj_pos', 'pipeline', fixed_params, buildAndTrainWSJPosModelWithUDPos, 'acc', 'ud_pos')
        if 'chunking' in config.tasks:
            run_build_model('chunking', 'pipeline', fixed_params, buildAndTrainChunkingModelWithNer, 'f1', 'ner')
            run_build_model('chunking', 'pipeline', fixed_params, buildAndTrainChunkingModelWithPos, 'f1', 'pos')
            run_build_model('chunking', 'pipeline', fixed_params, buildAndTrainChunkingModelWithNerPos, 'f1', 'pos-ner')
        if 'ud_pos' in config.tasks:
            run_build_model('ud_pos', 'pipeline', fixed_params, buildAndTrainUDPosModelWithWSJPos, 'acc', 'wsj_pos')

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

#run_models_as_input_exp_with_random_params()
#extendCoNLLNer()
#extendCoNLLChunking()
#extendWSJPOS()
#extendUDPOS()
run_models_as_input_exp_with_fixed_params()
