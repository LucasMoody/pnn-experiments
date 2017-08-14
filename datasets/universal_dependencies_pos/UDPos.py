from datasets import GermEvalReader, DatasetExtender
from keras.utils import np_utils
from os import path
import numpy as np

trainFile = 'datasets/universal_dependencies_pos/data/en-ud-train.conllu'
devFile = 'datasets/universal_dependencies_pos/data/en-ud-dev.conllu'
testFile = 'datasets/universal_dependencies_pos/data/en-ud-test.conllu'

pos_trainFileExt = 'datasets/universal_dependencies_pos/data/train_ext.conllu'
pos_devFileExt = 'datasets/universal_dependencies_pos/data/dev_ext.conllu'
pos_testFileExt = 'datasets/universal_dependencies_pos/data/test_ext.conllu'

directory = 'datasets/universal_dependencies_pos/data/'

word_position = 1
label_position = 3

ext_word_position = 0
ext_label_position = 1
ext_ner_position = 2
ext_chunking_position = 3
ext_wsj_pos_position = 4

def readDataset(windowSize, word2Idx, caseLookup):

    # Read in data
    print "Read in data and create matrices"
    pos_train_sentences = GermEvalReader.readFile(trainFile, word_position, label_position)
    pos_dev_sentences = GermEvalReader.readFile(devFile, word_position, label_position)
    pos_test_sentences = GermEvalReader.readFile(testFile, word_position, label_position)

    #Label mapping for POS
    pos_label2Idx, pos_idx2Label = GermEvalReader.getLabelDict(trainFile)

    # Create numpy arrays
    pos_train_x, pos_train_case_x, pos_train_y = GermEvalReader.createNumpyArrayWithCasing(pos_train_sentences, windowSize, word2Idx, pos_label2Idx, caseLookup)
    pos_dev_x, pos_dev_case_x, pos_dev_y = GermEvalReader.createNumpyArrayWithCasing(pos_dev_sentences, windowSize, word2Idx, pos_label2Idx, caseLookup)
    pos_test_x, pos_test_case_x, pos_test_y = GermEvalReader.createNumpyArrayWithCasing(pos_test_sentences, windowSize, word2Idx, pos_label2Idx, caseLookup)

    pos_input_train = [pos_train_x, pos_train_case_x]
    pos_input_dev = [pos_dev_x, pos_dev_case_x]
    pos_input_test = [pos_test_x, pos_test_case_x]

    pos_train_y_cat = np_utils.to_categorical(pos_train_y, len(pos_label2Idx))
    pos_dicts = [word2Idx, caseLookup, pos_label2Idx, pos_idx2Label]
    return [pos_input_train, pos_train_y_cat], [pos_input_dev, pos_dev_y], [pos_input_test, pos_test_y], pos_dicts

def readDatasetExt(windowSize, word2Idx, case2Idx):
    # load data
    pos_train_sentences = GermEvalReader.readFileExt(pos_trainFileExt)
    pos_dev_sentences = GermEvalReader.readFileExt(pos_devFileExt)
    pos_test_sentences = GermEvalReader.readFileExt(pos_testFileExt)

    # create dictionaries
    # Label mapping for POS
    label_column_train = filterColumn(pos_train_sentences, ext_label_position)
    word_column_train = filterColumn(pos_train_sentences, ext_word_position)
    ner_column_train = filterColumn(pos_train_sentences, ext_ner_position)
    chunking_column_train = filterColumn(pos_train_sentences, ext_chunking_position)
    wsj_pos_column_train = filterColumn(pos_train_sentences, ext_wsj_pos_position)

    label_column_dev = filterColumn(pos_dev_sentences, ext_label_position)
    word_column_dev = filterColumn(pos_dev_sentences, ext_word_position)
    ner_column_dev = filterColumn(pos_dev_sentences, ext_ner_position)
    chunking_column_dev = filterColumn(pos_dev_sentences, ext_chunking_position)
    wsj_pos_column_dev = filterColumn(pos_dev_sentences, ext_wsj_pos_position)

    label_column_test = filterColumn(pos_test_sentences, ext_label_position)
    word_column_test = filterColumn(pos_test_sentences, ext_word_position)
    ner_column_test = filterColumn(pos_test_sentences, ext_ner_position)
    chunking_column_test = filterColumn(pos_test_sentences, ext_chunking_position)
    wsj_pos_column_test = filterColumn(pos_test_sentences, ext_wsj_pos_position)

    pos_label2Idx, pos_idx2Label = DatasetExtender.getDict(label_column_train)
    pos_ner2Idx, pos_idx2ner = DatasetExtender.getDict(ner_column_train, withAddLabels=True)
    pos_chunking2Idx, pos_idx2chunking = DatasetExtender.getDict(chunking_column_train, withAddLabels=True)
    pos_wsj_pos2Idx, pos_idx2wsj_pos = DatasetExtender.getDict(wsj_pos_column_train, withAddLabels=True)

    # convert value to index
    words_train = GermEvalReader.convertValue2Idx(word_column_train, word2Idx, GermEvalReader.wordConverter)
    ner_train = GermEvalReader.convertValue2Idx(ner_column_train, pos_ner2Idx, GermEvalReader.wordConverter)
    chunking_train = GermEvalReader.convertValue2Idx(chunking_column_train, pos_chunking2Idx,
                                                     GermEvalReader.wordConverter)
    wsj_pos_train = GermEvalReader.convertValue2Idx(wsj_pos_column_train, pos_wsj_pos2Idx, GermEvalReader.wordConverter)
    casing_train = GermEvalReader.convertValue2Idx(word_column_train, case2Idx, GermEvalReader.getCasing)
    labels_train = GermEvalReader.convertValue2Idx(label_column_train, pos_label2Idx, GermEvalReader.labelConverter)

    words_dev = GermEvalReader.convertValue2Idx(word_column_dev, word2Idx, GermEvalReader.wordConverter)
    ner_dev = GermEvalReader.convertValue2Idx(ner_column_dev, pos_ner2Idx, GermEvalReader.wordConverter)
    chunking_dev = GermEvalReader.convertValue2Idx(chunking_column_dev, pos_chunking2Idx, GermEvalReader.wordConverter)
    wsj_pos_dev = GermEvalReader.convertValue2Idx(wsj_pos_column_dev, pos_wsj_pos2Idx, GermEvalReader.wordConverter)
    casing_dev = GermEvalReader.convertValue2Idx(word_column_dev, case2Idx, GermEvalReader.getCasing)
    labels_dev = GermEvalReader.convertValue2Idx(label_column_dev, pos_label2Idx, GermEvalReader.labelConverter)

    words_test = GermEvalReader.convertValue2Idx(word_column_test, word2Idx, GermEvalReader.wordConverter)
    ner_test = GermEvalReader.convertValue2Idx(ner_column_test, pos_ner2Idx, GermEvalReader.wordConverter)
    chunking_test = GermEvalReader.convertValue2Idx(chunking_column_test, pos_chunking2Idx,
                                                    GermEvalReader.wordConverter)
    wsj_pos_test = GermEvalReader.convertValue2Idx(wsj_pos_column_test, pos_wsj_pos2Idx, GermEvalReader.wordConverter)
    casing_test = GermEvalReader.convertValue2Idx(word_column_test, case2Idx, GermEvalReader.getCasing)
    labels_test = GermEvalReader.convertValue2Idx(label_column_test, pos_label2Idx, GermEvalReader.labelConverter)

    # create numpy datasets
    pos_train_x = GermEvalReader.createNumpyArray(words_train, windowSize, word2Idx)
    pos_train_ner_x = GermEvalReader.createNumpyArray(ner_train, windowSize, pos_ner2Idx)
    pos_train_chunking_x = GermEvalReader.createNumpyArray(chunking_train, windowSize, pos_chunking2Idx)
    pos_train_wsj_pos_x = GermEvalReader.createNumpyArray(wsj_pos_train, windowSize, pos_wsj_pos2Idx)
    pos_train_casing_x = GermEvalReader.createNumpyArray(casing_train, windowSize, case2Idx)
    pos_train_y = np.concatenate(labels_train)

    pos_dev_x = GermEvalReader.createNumpyArray(words_dev, windowSize, word2Idx)
    pos_dev_ner_x = GermEvalReader.createNumpyArray(ner_dev, windowSize, pos_ner2Idx)
    pos_dev_chunking_x = GermEvalReader.createNumpyArray(chunking_dev, windowSize, pos_chunking2Idx)
    pos_dev_wsj_pos_x = GermEvalReader.createNumpyArray(wsj_pos_dev, windowSize, pos_wsj_pos2Idx)
    pos_dev_casing_x = GermEvalReader.createNumpyArray(casing_dev, windowSize, case2Idx)
    pos_dev_y = np.concatenate(labels_dev)

    pos_test_x = GermEvalReader.createNumpyArray(words_test, windowSize, word2Idx)
    pos_test_ner_x = GermEvalReader.createNumpyArray(ner_test, windowSize, pos_ner2Idx)
    pos_test_chunking_x = GermEvalReader.createNumpyArray(chunking_test, windowSize, pos_chunking2Idx)
    pos_test_wsj_pos_x = GermEvalReader.createNumpyArray(wsj_pos_test, windowSize, pos_wsj_pos2Idx)
    pos_test_casing_x = GermEvalReader.createNumpyArray(casing_test, windowSize, case2Idx)
    pos_test_y = np.concatenate(labels_test)

    print "shape of pos_train_x:", pos_train_x.shape
    print pos_train_x[0]

    print "shape of pos_train_ner_x:", pos_train_ner_x.shape
    print pos_train_ner_x[0]

    print "shape of pos_train_chunking_x:", pos_train_chunking_x.shape
    print pos_train_chunking_x[0]

    print "shape of pos_train_wsj_pos_x:", pos_train_wsj_pos_x.shape
    print pos_train_wsj_pos_x[0]

    print "shape of pos_train_casing_x:", pos_train_casing_x.shape
    print pos_train_casing_x[0]

    print "shape of pos_train_y:", pos_train_y.shape
    print pos_train_y

    print "shape of pos_dev_x:", pos_dev_x.shape
    print pos_dev_x[0]

    print "shape of pos_dev_ner_x:", pos_dev_ner_x.shape
    print pos_dev_ner_x[0]

    print "shape of pos_dev_chunking_x:", pos_dev_chunking_x.shape
    print pos_dev_chunking_x[0]

    print "shape of pos_dev_wsj_pos_x:", pos_dev_wsj_pos_x.shape
    print pos_dev_wsj_pos_x[0]

    print "shape of pos_dev_casing_x:", pos_dev_casing_x.shape
    print pos_dev_casing_x[0]

    print "shape of pos_dev_y:", pos_dev_y.shape
    print pos_dev_y

    print "shape of pos_test_x:", pos_test_x.shape
    print pos_test_x[0]

    print "shape of pos_test_ner_x:", pos_test_ner_x.shape
    print pos_test_ner_x[0]

    print "shape of pos_test_chunking_x:", pos_test_chunking_x.shape
    print pos_test_chunking_x[0]

    print "shape of pos_test_wsj_pos_x:", pos_test_wsj_pos_x.shape
    print pos_test_wsj_pos_x[0]

    print "shape of pos_test_casing_x:", pos_test_casing_x.shape
    print pos_test_casing_x[0]

    print "shape of pos_test_y:", pos_test_y.shape
    print pos_test_y

    input_train = [pos_train_x, pos_train_casing_x, pos_train_ner_x, pos_train_chunking_x, pos_train_wsj_pos_x]
    input_dev = [pos_dev_x, pos_dev_casing_x, pos_dev_ner_x, pos_dev_chunking_x, pos_dev_wsj_pos_x]
    input_test = [pos_test_x, pos_test_casing_x, pos_test_ner_x, pos_test_chunking_x, pos_test_wsj_pos_x]

    pos_train_y_cat = np_utils.to_categorical(pos_train_y, len(pos_label2Idx))

    dicts = [word2Idx, case2Idx, pos_ner2Idx, pos_chunking2Idx, pos_wsj_pos2Idx, pos_label2Idx, pos_idx2Label]
    return [input_train, pos_train_y_cat], [input_dev, pos_dev_y], [input_test, pos_test_y], dicts

def filterColumn(sentences, position):
    return map(lambda sentence: sentence[:, position], sentences)

def extendDataset(train_extensions, dev_extensions, test_extensions):
    train_sentences = GermEvalReader.readFile(trainFile, word_position, label_position)
    dev_sentences = GermEvalReader.readFile(devFile, word_position, label_position)
    test_sentences = GermEvalReader.readFile(testFile, word_position, label_position)

    DatasetExtender.extendDataset("{0}train_ext.conllu".format(directory), train_sentences, train_extensions)
    DatasetExtender.extendDataset("{0}dev_ext.conllu".format(directory), dev_sentences, dev_extensions)
    DatasetExtender.extendDataset("{0}test_ext.conllu".format(directory), test_sentences, test_extensions)

def getLabelDict():
    return GermEvalReader.getLabelDict(trainFile)