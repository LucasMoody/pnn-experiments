from keras.layers import Input, Embedding, Flatten, merge

import embeddings.dependency_based_word_embeddings.DependencyBasedWordEmbeddings as Embeddings
from models import Trainer, InputBuilder
from datasets.conll_ner import CoNLLNer
from datasets.conll_chunking import CoNLLChunking
from datasets.wsj_pos import WSJPos
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

best_pos_window_size = 3
best_ud_pos_window_size = 3
best_ner_window_size = 3
best_chunking_window_size = 3

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
    # ----- read Data for pos with best pos window ----- #
    [input_train_for_pos, train_y_cat_for_pos], [input_dev_for_pos, dev_y_for_pos], [input_test_for_pos, test_y_for_pos], dicts_for_pos = CoNLLNer.readDataset(best_pos_window_size, word2Idx, case2Idx)

    [train_x_for_pos, train_case_x_for_pos] = input_train_for_pos
    n_in_x_for_pos = train_x_for_pos.shape[1]
    n_in_casing_for_pos = train_case_x_for_pos.shape[1]

    input_layers_for_pos, inputs_for_pos = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x_for_pos, n_in_casing_for_pos)

    pos_model = OptimizedModels.getWSJPOSModelGivenInput(input_layers_for_pos, inputs_for_pos, window_size=best_pos_window_size)

    #pos_model = OptimizedModels.getPOSModel(embeddings, word2Idx)
    pos_pred_train = pos_model.predict(input_train_for_pos, verbose=0).argmax(axis=-1)
    pos_pred_dev = pos_model.predict(input_dev_for_pos, verbose=0).argmax(axis=-1)
    pos_pred_test = pos_model.predict(input_test_for_pos, verbose=0).argmax(axis=-1)

    pos_label2Idx, pos_idx2Label = WSJPos.getLabelDict()
    pos_pred_train_labels = map(lambda idx: pos_idx2Label[idx], pos_pred_train)
    pos_pred_dev_labels = map(lambda idx: pos_idx2Label[idx], pos_pred_dev)
    pos_pred_test_labels = map(lambda idx: pos_idx2Label[idx], pos_pred_test)

    # ----- read Data for chunking with best chunking window ----- #
    [input_train_for_chunking, train_y_cat_for_chunking], [input_dev_for_chunking, dev_y_for_chunking], [input_test_for_chunking,
                                                                                     test_y_for_chunking], dicts_for_chunking = CoNLLNer.readDataset(
        best_chunking_window_size, word2Idx, case2Idx)

    [train_x_for_chunking, train_case_x_for_chunking] = input_train_for_chunking
    n_in_x_for_chunking = train_x_for_chunking.shape[1]
    n_in_casing_for_chunking = train_case_x_for_chunking.shape[1]

    input_layers_for_chunking, inputs_for_chunking = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x_for_chunking,
                                                                                n_in_casing_for_chunking)

    chunking_model = OptimizedModels.getChunkingModelGivenInput(input_layers_for_chunking, inputs_for_chunking,
                                                         window_size=best_chunking_window_size)

    chunking_pred_train = chunking_model.predict(input_train_for_chunking, verbose=0).argmax(axis=-1)
    chunking_pred_dev = chunking_model.predict(input_dev_for_chunking, verbose=0).argmax(axis=-1)
    chunking_pred_test = chunking_model.predict(input_test_for_chunking, verbose=0).argmax(axis=-1)

    chunking_label2Idx, chunking_idx2Label = CoNLLChunking.getLabelDict()
    chunking_pred_train_labels = map(lambda idx: chunking_idx2Label[idx], chunking_pred_train)
    chunking_pred_dev_labels = map(lambda idx: chunking_idx2Label[idx], chunking_pred_dev)
    chunking_pred_test_labels = map(lambda idx: chunking_idx2Label[idx], chunking_pred_test)

    train_extensions = [pos_pred_train_labels, chunking_pred_train_labels]
    dev_extensions = [pos_pred_dev_labels, chunking_pred_dev_labels]
    test_extensions = [pos_pred_test_labels, chunking_pred_test_labels]

    CoNLLNer.extendDataset("./datasets/conll_ner/data/eng.conllu", train_extensions, dev_extensions, test_extensions)

def extendUDPOS():
    # ----- read Data for ner with best ner window ----- #
    [input_train_for_ner, train_y_cat_for_ner], [input_dev_for_ner, dev_y_for_ner], [input_test_for_ner, test_y_for_ner] = UDPos.readDataset(best_ner_window_size, word2Idx, case2Idx)

    [train_x_for_ner, train_case_x_for_ner] = input_train_for_ner
    n_in_x_for_ner = train_x_for_ner.shape[1]
    n_in_casing_for_ner = train_case_x_for_ner.shape[1]

    input_layers_for_ner, inputs_for_ner = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x_for_ner, n_in_casing_for_ner)

    ner_model = OptimizedModels.getNERModelGivenInput(input_layers_for_ner, inputs_for_ner,
                                                            window_size=best_ner_window_size)
    ner_pred_train = ner_model.predict(input_train_for_ner, verbose=0).argmax(axis=-1)
    ner_pred_dev = ner_model.predict(input_dev_for_ner, verbose=0).argmax(axis=-1)
    ner_pred_test = ner_model.predict(input_test_for_ner, verbose=0).argmax(axis=-1)

    ner_label2Idx, ner_idx2Label = CoNLLNer.getLabelDict()
    ner_pred_train_labels = map(lambda idx: ner_idx2Label[idx], ner_pred_train)
    ner_pred_dev_labels = map(lambda idx: ner_idx2Label[idx], ner_pred_dev)
    ner_pred_test_labels = map(lambda idx: ner_idx2Label[idx], ner_pred_test)

    # ----- read Data for chunking with best chunking window ----- #
    [input_train_for_chunking, train_y_cat_for_chunking], [input_dev_for_chunking, dev_y_for_chunking], [input_test_for_chunking,
                                                                                     test_y_for_chunking] = UDPos.readDataset(
        best_chunking_window_size, word2Idx, case2Idx)

    [train_x_for_chunking, train_case_x_for_chunking] = input_train_for_chunking
    n_in_x_for_chunking = train_x_for_chunking.shape[1]
    n_in_casing_for_chunking = train_case_x_for_chunking.shape[1]

    input_layers_for_chunking, inputs_for_chunking = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x_for_chunking,
                                                                                n_in_casing_for_chunking)

    chunking_model = OptimizedModels.getChunkingModelGivenInput(input_layers_for_chunking, inputs_for_chunking,
                                                      window_size=best_chunking_window_size)
    chunking_pred_train = chunking_model.predict(input_train_for_chunking, verbose=0).argmax(axis=-1)
    chunking_pred_dev = chunking_model.predict(input_dev_for_chunking, verbose=0).argmax(axis=-1)
    chunking_pred_test = chunking_model.predict(input_test_for_chunking, verbose=0).argmax(axis=-1)

    chunking_label2Idx, chunking_idx2Label = CoNLLChunking.getLabelDict()
    chunking_pred_train_labels = map(lambda idx: chunking_idx2Label[idx], chunking_pred_train)
    chunking_pred_dev_labels = map(lambda idx: chunking_idx2Label[idx], chunking_pred_dev)
    chunking_pred_test_labels = map(lambda idx: chunking_idx2Label[idx], chunking_pred_test)

    # ----- read Data for pos with best wsj_pos window ----- #
    [input_train_for_wsj_pos, train_y_cat_for_wsj_pos], [input_dev_for_wsj_pos, dev_y_for_wsj_pos], [
        input_test_for_wsj_pos,
        test_y_for_wsj_pos] = UDPos.readDataset(
        best_pos_window_size, word2Idx, case2Idx)

    [train_x_for_wsj_pos, train_case_x_for_wsj_pos] = input_train_for_wsj_pos
    n_in_x_for_wsj_pos = train_x_for_wsj_pos.shape[1]
    n_in_casing_for_wsj_pos = train_case_x_for_wsj_pos.shape[1]

    input_layers_for_wsj_pos, inputs_for_wsj_pos = InputBuilder.buildStandardModelInput(embeddings, case2Idx,
                                                                                          n_in_x_for_wsj_pos,
                                                                                          n_in_casing_for_wsj_pos)

    wsj_pos_model = OptimizedModels.getWSJPOSModelGivenInput(input_layers_for_wsj_pos, inputs_for_wsj_pos,
                                                                window_size=best_pos_window_size)
    wsj_pos_pred_train = wsj_pos_model.predict(input_train_for_wsj_pos, verbose=0).argmax(axis=-1)
    wsj_pos_pred_dev = wsj_pos_model.predict(input_dev_for_wsj_pos, verbose=0).argmax(axis=-1)
    wsj_pos_pred_test = wsj_pos_model.predict(input_test_for_wsj_pos, verbose=0).argmax(axis=-1)

    wsj_pos_label2Idx, wsj_pos_idx2Label = WSJPos.getLabelDict()
    wsj_pos_pred_train_labels = map(lambda idx: wsj_pos_idx2Label[idx], wsj_pos_pred_train)
    wsj_pos_pred_dev_labels = map(lambda idx: wsj_pos_idx2Label[idx], wsj_pos_pred_dev)
    wsj_pos_pred_test_labels = map(lambda idx: wsj_pos_idx2Label[idx], wsj_pos_pred_test)

    train_extensions = [ner_pred_train_labels, chunking_pred_train_labels, wsj_pos_pred_train_labels]
    dev_extensions = [ner_pred_dev_labels, chunking_pred_dev_labels, wsj_pos_pred_dev_labels]
    test_extensions = [ner_pred_test_labels, chunking_pred_test_labels, wsj_pos_pred_test_labels]

    UDPos.extendDataset("./datasets/universal_dependencies_pos/data/en-ud.conllu", train_extensions, dev_extensions, test_extensions)

def extendWSJPOS():
    # ----- read Data for ner with best ner window ----- #
    [input_train_for_ner, train_y_cat_for_ner], [input_dev_for_ner, dev_y_for_ner], [input_test_for_ner, test_y_for_ner] = WSJPos.readDataset(best_ner_window_size, word2Idx, case2Idx)

    [train_x_for_ner, train_case_x_for_ner] = input_train_for_ner
    n_in_x_for_ner = train_x_for_ner.shape[1]
    n_in_casing_for_ner = train_case_x_for_ner.shape[1]

    input_layers_for_ner, inputs_for_ner = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x_for_ner, n_in_casing_for_ner)

    ner_model = OptimizedModels.getNERModelGivenInput(input_layers_for_ner, inputs_for_ner,
                                                            window_size=best_ner_window_size)
    ner_pred_train = ner_model.predict(input_train_for_ner, verbose=0).argmax(axis=-1)
    ner_pred_dev = ner_model.predict(input_dev_for_ner, verbose=0).argmax(axis=-1)
    ner_pred_test = ner_model.predict(input_test_for_ner, verbose=0).argmax(axis=-1)

    ner_label2Idx, ner_idx2Label = CoNLLNer.getLabelDict()
    ner_pred_train_labels = map(lambda idx: ner_idx2Label[idx], ner_pred_train)
    ner_pred_dev_labels = map(lambda idx: ner_idx2Label[idx], ner_pred_dev)
    ner_pred_test_labels = map(lambda idx: ner_idx2Label[idx], ner_pred_test)

    # ----- read Data for chunking with best chunking window ----- #
    [input_train_for_chunking, train_y_cat_for_chunking], [input_dev_for_chunking, dev_y_for_chunking], [
        input_test_for_chunking,
        test_y_for_chunking] = WSJPos.readDataset(
        best_chunking_window_size, word2Idx, case2Idx)

    [train_x_for_chunking, train_case_x_for_chunking] = input_train_for_chunking
    n_in_x_for_chunking = train_x_for_chunking.shape[1]
    n_in_casing_for_chunking = train_case_x_for_chunking.shape[1]

    input_layers_for_chunking, inputs_for_chunking = InputBuilder.buildStandardModelInput(embeddings, case2Idx,
                                                                                          n_in_x_for_chunking,
                                                                                          n_in_casing_for_chunking)

    chunking_model = OptimizedModels.getChunkingModelGivenInput(input_layers_for_chunking, inputs_for_chunking,
                                                                window_size=best_chunking_window_size)
    chunking_pred_train = chunking_model.predict(input_train_for_chunking, verbose=0).argmax(axis=-1)
    chunking_pred_dev = chunking_model.predict(input_dev_for_chunking, verbose=0).argmax(axis=-1)
    chunking_pred_test = chunking_model.predict(input_test_for_chunking, verbose=0).argmax(axis=-1)

    chunking_label2Idx, chunking_idx2Label = CoNLLChunking.getLabelDict()
    chunking_pred_train_labels = map(lambda idx: chunking_idx2Label[idx], chunking_pred_train)
    chunking_pred_dev_labels = map(lambda idx: chunking_idx2Label[idx], chunking_pred_dev)
    chunking_pred_test_labels = map(lambda idx: chunking_idx2Label[idx], chunking_pred_test)

    # ----- read Data for pos with best ud_pos window ----- #
    [input_train_for_ud_pos, train_y_cat_for_ud_pos], [input_dev_for_ud_pos, dev_y_for_ud_pos], [
        input_test_for_ud_pos,
        test_y_for_ud_pos] = WSJPos.readDataset(
        best_ud_pos_window_size, word2Idx, case2Idx)

    [train_x_for_ud_pos, train_case_x_for_ud_pos] = input_train_for_ud_pos
    n_in_x_for_ud_pos = train_x_for_ud_pos.shape[1]
    n_in_casing_for_ud_pos = train_case_x_for_ud_pos.shape[1]

    input_layers_for_ud_pos, inputs_for_ud_pos = InputBuilder.buildStandardModelInput(embeddings, case2Idx,
                                                                                          n_in_x_for_ud_pos,
                                                                                          n_in_casing_for_ud_pos)

    ud_pos_model = OptimizedModels.getUDPOSModelGivenInput(input_layers_for_ud_pos, inputs_for_ud_pos,
                                                                window_size=best_ud_pos_window_size)
    ud_pos_pred_train = ud_pos_model.predict(input_train_for_ud_pos, verbose=0).argmax(axis=-1)
    ud_pos_pred_dev = ud_pos_model.predict(input_dev_for_ud_pos, verbose=0).argmax(axis=-1)
    ud_pos_pred_test = ud_pos_model.predict(input_test_for_ud_pos, verbose=0).argmax(axis=-1)

    ud_pos_label2Idx, ud_pos_idx2Label = UDPos.getLabelDict()
    ud_pos_pred_train_labels = map(lambda idx: ud_pos_idx2Label[idx], ud_pos_pred_train)
    ud_pos_pred_dev_labels = map(lambda idx: ud_pos_idx2Label[idx], ud_pos_pred_dev)
    ud_pos_pred_test_labels = map(lambda idx: ud_pos_idx2Label[idx], ud_pos_pred_test)

    train_extensions = [ner_pred_train_labels, chunking_pred_train_labels, ud_pos_pred_train_labels]
    dev_extensions = [ner_pred_dev_labels, chunking_pred_dev_labels, ud_pos_pred_dev_labels]
    test_extensions = [ner_pred_test_labels, chunking_pred_test_labels, ud_pos_pred_test_labels]

    WSJPos.extendDataset("./datasets/wsj_pos/data/wsj.conllu", train_extensions, dev_extensions, test_extensions)

def extendCoNLLChunking():
    # ----- read Data for pos with best pos window ----- #
    [input_train_for_pos, train_y_cat_for_pos], [input_dev_for_pos, dev_y_for_pos], [input_test_for_pos, test_y_for_pos], dicts_for_pos = CoNLLChunking.readDataset(best_pos_window_size, word2Idx, case2Idx)

    # calculate dims for model building
    [train_x_for_pos, train_case_x_for_pos] = input_train_for_pos
    n_in_x_for_pos = train_x_for_pos.shape[1]
    n_in_casing_for_pos = train_case_x_for_pos.shape[1]

    # build pos model
    input_layers_for_pos, inputs_for_pos = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x_for_pos, n_in_casing_for_pos)
    pos_model = OptimizedModels.getWSJPOSModelGivenInput(input_layers_for_pos, inputs_for_pos, window_size=best_pos_window_size)

    # predict pos on chunking data
    pos_pred_train = pos_model.predict(input_train_for_pos, verbose=0).argmax(axis=-1)
    pos_pred_dev = pos_model.predict(input_dev_for_pos, verbose=0).argmax(axis=-1)
    pos_pred_test = pos_model.predict(input_test_for_pos, verbose=0).argmax(axis=-1)

    #
    pos_label2Idx, pos_idx2Label = WSJPos.getLabelDict()
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
    ner_model = OptimizedModels.getNERModelGivenInput(input_layers_for_ner, inputs_for_ner, window_size=best_ner_window_size)

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

def buildAndTrainNERModelWithChunkingPos(learning_params=None):
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

    model = NER.buildNERModelGivenInput(input_layers_merged, inputs, params, ner_n_out)

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(ner_idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, input_train,
                                                                   ner_train_y_cat, number_of_epochs,
                                                                   params['batch_size'], input_dev,
                                                                   ner_dev_y, input_test, ner_test_y,
                                                                   measurements=[biof1])

    return train_scores, dev_scores, test_scores

def buildAndTrainNERModelWithChunking(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train, ner_train_y_cat], [input_dev, ner_dev_y], [input_test, ner_test_y], dicts = CoNLLNer.readDatasetExt(params['window_size'], word2Idx, case2Idx)

    [ner_train_x, ner_train_pos_x, ner_train_chunking_x, ner_train_casing_x] = input_train
    [ner_dev_x, ner_dev_pos_x, ner_dev_chunking_x, ner_dev_casing_x] = input_dev
    [ner_test_x, ner_test_pos_x, ner_test_chunking_x, ner_test_casing_x] = input_test
    [_, pos2Idx, chunking2Idx, _, ner_label2Idx, ner_idx2Label] = dicts

    model_input_train = [ner_train_x, ner_train_chunking_x, ner_train_casing_x]
    model_input_dev = [ner_dev_x, ner_dev_chunking_x, ner_dev_casing_x]
    model_input_test = [ner_test_x, ner_test_chunking_x, ner_test_casing_x]

    ner_n_out = ner_train_y_cat.shape[1]
    n_in_x = ner_train_x.shape[1]
    n_in_chunking = ner_train_chunking_x.shape[1]
    n_in_casing = ner_train_casing_x.shape[1]


    words_input = Input(shape=(n_in_x,), dtype='int32', name='words_input')
    wordEmbeddingLayer = Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in_x,
                                   weights=[embeddings], trainable=False)
    words = wordEmbeddingLayer(words_input)
    words = Flatten(name='words_flatten')(words)

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

    input_layers = [words, chunking, casing]
    inputs = [words_input, chunking_input, case_input]

    input_layers_merged = merge(input_layers, mode='concat')

    model = NER.buildNERModelGivenInput(input_layers_merged, inputs, params, ner_n_out)

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(ner_idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, model_input_train,
                                                                   ner_train_y_cat, number_of_epochs,
                                                                   params['batch_size'], model_input_dev,
                                                                   ner_dev_y, model_input_test, ner_test_y,
                                                                   measurements=[biof1])

    return train_scores, dev_scores, test_scores

def buildAndTrainNERModelWithPos(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train, ner_train_y_cat], [input_dev, ner_dev_y], [input_test, ner_test_y], dicts = CoNLLNer.readDatasetExt(params['window_size'], word2Idx, case2Idx)

    [ner_train_x, ner_train_pos_x, ner_train_chunking_x, ner_train_casing_x] = input_train
    [ner_dev_x, ner_dev_pos_x, ner_dev_chunking_x, ner_dev_casing_x] = input_dev
    [ner_test_x, ner_test_pos_x, ner_test_chunking_x, ner_test_casing_x] = input_test
    [_, pos2Idx, chunking2Idx, _, ner_label2Idx, ner_idx2Label] = dicts

    model_input_train = [ner_train_x, ner_train_pos_x, ner_train_casing_x]
    model_input_dev = [ner_dev_x, ner_dev_pos_x, ner_dev_casing_x]
    model_input_test = [ner_test_x, ner_test_pos_x, ner_test_casing_x]

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
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, model_input_train,
                                                                   ner_train_y_cat, number_of_epochs,
                                                                   params['batch_size'], model_input_dev,
                                                                   ner_dev_y, model_input_test, ner_test_y,
                                                                   measurements=[biof1])

    return train_scores, dev_scores, test_scores

def buildAndTrainChunkingModelWithNerPos(learning_params=None):
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

def buildAndTrainChunkingModelWithNer(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = CoNLLChunking.readDatasetExt(params['window_size'], word2Idx, case2Idx)

    [chunking_train_x, chunking_train_pos_x, chunking_train_ner_x, chunking_train_casing_x] = input_train
    [chunking_dev_x, chunking_dev_pos_x, chunking_dev_ner_x, chunking_dev_casing_x] = input_dev
    [chunking_test_x, chunking_test_pos_x, chunking_test_ner_x, chunking_test_casing_x] = input_test
    [_, pos2Idx, ner2Idx, _, chunking_label2Idx, chunking_idx2Label] = dicts

    model_input_train = [chunking_train_x, chunking_train_ner_x, chunking_train_casing_x]
    model_input_dev = [chunking_dev_x, chunking_dev_ner_x, chunking_dev_casing_x]
    model_input_test = [chunking_test_x, chunking_test_ner_x, chunking_test_casing_x]

    chunking_n_out = train_y_cat.shape[1]
    n_in_x = chunking_train_x.shape[1]
    n_in_ner = chunking_train_ner_x.shape[1]
    n_in_casing = chunking_train_casing_x.shape[1]

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

    model = Chunking.buildChunkingModelGivenInput(input_layers_merged, inputs, params, chunking_n_out)

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(chunking_idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, model_input_train,
                                                                   train_y_cat, number_of_epochs,
                                                                   params['batch_size'], model_input_dev,
                                                                   dev_y, model_input_test, test_y,
                                                                   measurements=[biof1])

    return train_scores, dev_scores, test_scores


def buildAndTrainChunkingModelWithPos(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = CoNLLChunking.readDatasetExt(
        params['window_size'], word2Idx, case2Idx)

    [chunking_train_x, chunking_train_pos_x, chunking_train_ner_x, chunking_train_casing_x] = input_train
    [chunking_dev_x, chunking_dev_pos_x, chunking_dev_ner_x, chunking_dev_casing_x] = input_dev
    [chunking_test_x, chunking_test_pos_x, chunking_test_ner_x, chunking_test_casing_x] = input_test
    [_, pos2Idx, ner2Idx, _, chunking_label2Idx, chunking_idx2Label] = dicts

    model_input_train = [chunking_train_x, chunking_train_pos_x, chunking_train_casing_x]
    model_input_dev = [chunking_dev_x, chunking_dev_pos_x, chunking_dev_casing_x]
    model_input_test = [chunking_test_x, chunking_test_pos_x, chunking_test_casing_x]

    chunking_n_out = train_y_cat.shape[1]
    n_in_x = chunking_train_x.shape[1]
    n_in_pos = chunking_train_pos_x.shape[1]
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

    case_input = Input(shape=(n_in_x,), dtype='int32', name='case_input')
    caseEmbeddingLayer = Embedding(output_dim=len(case2Idx), input_dim=len(case2Idx), input_length=n_in_casing,
                                   trainable=True)
    casing = caseEmbeddingLayer(case_input)
    casing = Flatten(name='casing_flatten')(casing)

    input_layers = [words, pos, casing]
    inputs = [words_input, pos_input, case_input]

    input_layers_merged = merge(input_layers, mode='concat')

    model = Chunking.buildChunkingModelGivenInput(input_layers_merged, inputs, params, chunking_n_out)

    # ----- Train Model ----- #
    biof1 = Measurer.create_compute_BIOf1(chunking_idx2Label)
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, model_input_train,
                                                                                 train_y_cat, number_of_epochs,
                                                                                 params['batch_size'], model_input_dev,
                                                                                 dev_y, model_input_test, test_y,
                                                                                 measurements=[biof1])

    return train_scores, dev_scores, test_scores


def buildAndTrainPOSModelWithChunkingNer(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = WSJPos.readDatasetExt(params['window_size'], word2Idx, case2Idx)

    [pos_train_x, pos_train_ner_x, pos_train_chunking_x, pos_train_ud_pos_x, pos_train_casing_x] = input_train
    [pos_dev_x, pos_dev_ner_x, pos_dev_chunking_x, pos_dev_ud_pos_x, pos_dev_casing_x] = input_dev
    [pos_test_x, pos_test_ner_x, pos_test_chunking_x, pos_test_ud_pos_x, pos_test_casing_x] = input_test
    [_, ner2Idx, chunking2Idx, ud_pos2Idx, _, label2Idx, idx2Label] = dicts

    model_input_train = [pos_train_x, pos_train_ner_x, pos_train_chunking_x, pos_train_casing_x]
    model_input_dev = [pos_dev_x, pos_dev_ner_x, pos_dev_chunking_x, pos_dev_casing_x]
    model_input_test = [pos_test_x, pos_test_ner_x, pos_test_chunking_x, pos_test_casing_x]

    pos_n_out = train_y_cat.shape[1]
    n_in_x = pos_train_x.shape[1]
    n_in_ner = pos_train_ner_x.shape[1]
    n_in_chunking = pos_train_chunking_x.shape[1]
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

    input_layers = [words, ner, chunking, casing]
    inputs = [words_input, ner_input, chunking_input, case_input]

    input_layers_merged = merge(input_layers, mode='concat')

    model = POS.buildPosModelGivenInput(input_layers_merged, inputs, params, pos_n_out)

    # ----- Train Model ----- #
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, model_input_train,
                                                                   train_y_cat, number_of_epochs,
                                                                   params['batch_size'], model_input_dev,
                                                                   dev_y, model_input_test, test_y,
                                                                   measurements=[Measurer.measureAccuracy])

    return train_scores, dev_scores, test_scores

def buildAndTrainPOSModelWithChunking(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = WSJPos.readDatasetExt(params['window_size'], word2Idx, case2Idx)

    [pos_train_x, pos_train_ner_x, pos_train_chunking_x, pos_train_ud_pos_x, pos_train_casing_x] = input_train
    [pos_dev_x, pos_dev_ner_x, pos_dev_chunking_x, pos_dev_ud_pos_x, pos_dev_casing_x] = input_dev
    [pos_test_x, pos_test_ner_x, pos_test_chunking_x, pos_test_ud_pos_x, pos_test_casing_x] = input_test
    [_, ner2Idx, chunking2Idx, ud_pos2Idx, _, label2Idx, idx2Label] = dicts

    model_input_train = [pos_train_x, pos_train_chunking_x, pos_train_casing_x]
    model_input_dev = [pos_dev_x, pos_dev_chunking_x, pos_dev_casing_x]
    model_input_test = [pos_test_x, pos_test_chunking_x, pos_test_casing_x]

    pos_n_out = train_y_cat.shape[1]
    n_in_x = pos_train_x.shape[1]
    n_in_chunking = pos_train_chunking_x.shape[1]
    n_in_casing = pos_train_casing_x.shape[1]

    words_input = Input(shape=(n_in_x,), dtype='int32', name='words_input')
    wordEmbeddingLayer = Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in_x,
                                   weights=[embeddings], trainable=False)
    words = wordEmbeddingLayer(words_input)
    words = Flatten(name='words_flatten')(words)

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

    input_layers = [words, chunking, casing]
    inputs = [words_input, chunking_input, case_input]

    input_layers_merged = merge(input_layers, mode='concat')

    model = POS.buildPosModelGivenInput(input_layers_merged, inputs, params, pos_n_out)

    # ----- Train Model ----- #
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, model_input_train,
                                                                   train_y_cat, number_of_epochs,
                                                                   params['batch_size'], model_input_dev,
                                                                   dev_y, model_input_test, test_y,
                                                                   measurements=[Measurer.measureAccuracy])

    return train_scores, dev_scores, test_scores

def buildAndTrainPOSModelWithNer(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = WSJPos.readDatasetExt(params['window_size'], word2Idx, case2Idx)

    [pos_train_x, pos_train_ner_x, pos_train_chunking_x, pos_train_ud_pos_x, pos_train_casing_x] = input_train
    [pos_dev_x, pos_dev_ner_x, pos_dev_chunking_x, pos_dev_ud_pos_x, pos_dev_casing_x] = input_dev
    [pos_test_x, pos_test_ner_x, pos_test_chunking_x, pos_test_ud_pos_x, pos_test_casing_x] = input_test
    [_, ner2Idx, chunking2Idx, ud_pos2Idx, _, label2Idx, idx2Label] = dicts

    model_input_train = [pos_train_x, pos_train_ner_x, pos_train_casing_x]
    model_input_dev = [pos_dev_x, pos_dev_ner_x, pos_dev_casing_x]
    model_input_test = [pos_test_x, pos_test_ner_x, pos_test_casing_x]

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
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, model_input_train,
                                                                   train_y_cat, number_of_epochs,
                                                                   params['batch_size'], model_input_dev,
                                                                   dev_y, model_input_test, test_y,
                                                                   measurements=[Measurer.measureAccuracy])

    return train_scores, dev_scores, test_scores

def buildAndTrainPOSModelWithUDPos(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = WSJPos.readDatasetExt(params['window_size'], word2Idx, case2Idx)

    [pos_train_x, pos_train_ner_x, pos_train_chunking_x, pos_train_ud_pos_x, pos_train_casing_x] = input_train
    [pos_dev_x, pos_dev_ner_x, pos_dev_chunking_x, pos_dev_ud_pos_x, pos_dev_casing_x] = input_dev
    [pos_test_x, pos_test_ner_x, pos_test_chunking_x, pos_test_ud_pos_x, pos_test_casing_x] = input_test
    [_, ner2Idx, chunking2Idx, ud_pos2Idx, _, label2Idx, idx2Label] = dicts

    model_input_train = [pos_train_x, pos_train_ud_pos_x, pos_train_casing_x]
    model_input_dev = [pos_dev_x, pos_dev_ud_pos_x, pos_dev_casing_x]
    model_input_test = [pos_test_x, pos_test_ud_pos_x, pos_test_casing_x]

    pos_n_out = train_y_cat.shape[1]
    n_in_x = pos_train_x.shape[1]
    n_in_ud_pos = pos_train_ud_pos_x.shape[1]
    n_in_casing = pos_train_casing_x.shape[1]

    words_input = Input(shape=(n_in_x,), dtype='int32', name='words_input')
    wordEmbeddingLayer = Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in_x,
                                   weights=[embeddings], trainable=False)
    words = wordEmbeddingLayer(words_input)
    words = Flatten(name='words_flatten')(words)

    ud_pos_input = Input(shape=(n_in_ud_pos,), dtype='int32', name='ud_pos_input')
    ud_posEmbeddingLayer = Embedding(output_dim=len(ud_pos2Idx), input_dim=len(ud_pos2Idx), input_length=n_in_ud_pos,
                                   trainable=True)
    ud_pos = ud_posEmbeddingLayer(ud_pos_input)
    ud_pos = Flatten(name='ud_pos_flatten')(ud_pos)

    case_input = Input(shape=(n_in_x,), dtype='int32', name='case_input')
    caseEmbeddingLayer = Embedding(output_dim=len(case2Idx), input_dim=len(case2Idx), input_length=n_in_casing,
                                   trainable=True)
    casing = caseEmbeddingLayer(case_input)
    casing = Flatten(name='casing_flatten')(casing)

    input_layers = [words, ud_pos, casing]
    inputs = [words_input, ud_pos_input, case_input]

    input_layers_merged = merge(input_layers, mode='concat')

    model = POS.buildPosModelGivenInput(input_layers_merged, inputs, params, pos_n_out)

    # ----- Train Model ----- #
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, model_input_train,
                                                                   train_y_cat, number_of_epochs,
                                                                   params['batch_size'], model_input_dev,
                                                                   dev_y, model_input_test, test_y,
                                                                   measurements=[Measurer.measureAccuracy])

    return train_scores, dev_scores, test_scores

def buildAndTrainUDPosModelWithWSJPos(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = UDPos.readDatasetExt(params['window_size'], word2Idx, case2Idx)

    [pos_train_x, pos_train_ner_x, pos_train_chunking_x, pos_train_wsj_pos_x, pos_train_casing_x] = input_train
    [pos_dev_x, pos_dev_ner_x, pos_dev_chunking_x, pos_dev_wsj_pos_x, pos_dev_casing_x] = input_dev
    [pos_test_x, pos_test_ner_x, pos_test_chunking_x, pos_test_wsj_pos_x, pos_test_casing_x] = input_test
    [_, ner2Idx, chunking2Idx, wsj_pos2Idx, _, label2Idx, idx2Label] = dicts

    model_input_train = [pos_train_x, pos_train_wsj_pos_x, pos_train_casing_x]
    model_input_dev = [pos_dev_x, pos_dev_wsj_pos_x, pos_dev_casing_x]
    model_input_test = [pos_test_x, pos_test_wsj_pos_x, pos_test_casing_x]

    pos_n_out = train_y_cat.shape[1]
    n_in_x = pos_train_x.shape[1]
    n_in_wsj_pos = pos_train_wsj_pos_x.shape[1]
    n_in_casing = pos_train_casing_x.shape[1]

    words_input = Input(shape=(n_in_x,), dtype='int32', name='words_input')
    wordEmbeddingLayer = Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in_x,
                                   weights=[embeddings], trainable=False)
    words = wordEmbeddingLayer(words_input)
    words = Flatten(name='words_flatten')(words)

    wsj_pos_input = Input(shape=(n_in_wsj_pos,), dtype='int32', name='wsj_pos_input')
    wsj_posEmbeddingLayer = Embedding(output_dim=len(wsj_pos2Idx), input_dim=len(wsj_pos2Idx), input_length=n_in_wsj_pos,
                                   trainable=True)
    wsj_pos = wsj_posEmbeddingLayer(wsj_pos_input)
    wsj_pos = Flatten(name='wsj_pos_flatten')(wsj_pos)

    case_input = Input(shape=(n_in_x,), dtype='int32', name='case_input')
    caseEmbeddingLayer = Embedding(output_dim=len(case2Idx), input_dim=len(case2Idx), input_length=n_in_casing,
                                   trainable=True)
    casing = caseEmbeddingLayer(case_input)
    casing = Flatten(name='casing_flatten')(casing)

    input_layers = [words, wsj_pos, casing]
    inputs = [words_input, wsj_pos_input, case_input]

    input_layers_merged = merge(input_layers, mode='concat')

    model = POS.buildPosModelGivenInput(input_layers_merged, inputs, params, pos_n_out)

    # ----- Train Model ----- #
    train_scores, dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model, model_input_train,
                                                                   train_y_cat, number_of_epochs,
                                                                   params['batch_size'], model_input_dev,
                                                                   dev_y, model_input_test, test_y,
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
        best_train_scores_ner, best_dev_scores_ner, best_test_scores_ner = buildAndTrainNERModelWithChunkingPos(params)
        print params
        for (sample_scores, sample) in best_train_scores_ner:
            for score in sample_scores:
                print "Max f1 train ner: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'ner', 'train', params, score[0], score[1], sample, 'pos')
        for (sample_scores, sample) in best_dev_scores_ner:
            for score in sample_scores:
                print "Max f1 dev ner: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'ner', 'dev', params, score[0], score[1], sample, 'pos')
        for (sample_scores, sample) in best_test_scores_ner:
            for score in sample_scores:
                print "Max f1 test ner: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'ner', 'test', params, score[0], score[1], sample, 'pos')

        best_train_scores_chunking, best_dev_scores_chunking, best_test_scores_chunking = buildAndTrainChunkingModelWithNerPos(params)
        print params
        for (sample_scores, sample) in best_train_scores_chunking:
            for score in sample_scores:
                print "Max f1 train chunking: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'chunking', 'train', params,
                                                     score[0], score[1], sample, 'pos-ner')
        for (sample_scores, sample) in best_dev_scores_chunking:
            for score in sample_scores:
                print "Max f1 dev chunking: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'chunking', 'dev', params,
                                                     score[0], score[1], sample, 'pos-ner')
        for (sample_scores, sample) in best_test_scores_chunking:
            for score in sample_scores:
                print "Max f1 test chunking: %.4f in epoch: %d with samples: %d" % (score[0], sample, score[1])
                Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_2', 'chunking', 'test', params,
                                                     score[0], score[1], sample, 'pos-ner')

        best_train_scores_pos, best_dev_scores_pos, best_test_scores_pos = buildAndTrainPOSModelWithChunkingNer(params)
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
        'activation': 'tanh',
        'dropout': 0.3,
        'optimizer': 'adam',
        'number_of_epochs': config.number_of_epochs
    }
    max_evals = config.number_of_evals
    for model_nr in range(max_evals):
        print "Model nr. ", model_nr

        if 'ner' in config.tasks:
            run_build_model('ner', 'exp_2', fixed_params, buildAndTrainNERModelWithPos, 'f1', 'pos')
            run_build_model('ner', 'exp_2', fixed_params, buildAndTrainNERModelWithChunking, 'f1', 'chunking')
            run_build_model('ner', 'exp_2', fixed_params, buildAndTrainNERModelWithChunkingPos, 'f1', 'chunking-pos')
        if 'wsj_pos' in config.tasks:
            run_build_model('wsj_pos', 'exp_2', fixed_params, buildAndTrainPOSModelWithNer, 'acc', 'ner')
            run_build_model('wsj_pos', 'exp_2', fixed_params, buildAndTrainPOSModelWithChunking, 'acc', 'chunking')
            run_build_model('wsj_pos', 'exp_2', fixed_params, buildAndTrainPOSModelWithChunkingNer, 'acc', 'chunking-ner')
            run_build_model('wsj_pos', 'exp_2', fixed_params, buildAndTrainPOSModelWithUDPos, 'acc', 'ud_pos')
        if 'chunking' in config.tasks:
            run_build_model('chunking', 'exp_2', fixed_params, buildAndTrainChunkingModelWithNer, 'f1', 'ner')
            run_build_model('chunking', 'exp_2', fixed_params, buildAndTrainChunkingModelWithPos, 'f1', 'pos')
            run_build_model('chunking', 'exp_2', fixed_params, buildAndTrainChunkingModelWithNerPos, 'f1', 'pos-ner')
        if 'ud_pos' in config.tasks:
            run_build_model('ud_pos', 'exp_2', fixed_params, buildAndTrainUDPosModelWithWSJPos, 'acc', 'wsj_pos')

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
