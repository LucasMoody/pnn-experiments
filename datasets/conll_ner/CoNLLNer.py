from datasets import GermEvalReader, DatasetExtender
from keras.utils import np_utils
from os import path
import numpy as np

ner_trainFile = 'datasets/conll_ner/data/eng.train'
ner_devFile = 'datasets/conll_ner/data/eng.testa'
ner_testFile = 'datasets/conll_ner/data/eng.testb'

ner_trainFileExt = 'datasets/conll_ner/data/eng_train_ext.conllu'
ner_devFileExt = 'datasets/conll_ner/data/eng_dev_ext.conllu'
ner_testFileExt = 'datasets/conll_ner/data/eng_test_ext.conllu'

word_position = 0
label_position = 1
pos_position = 2
other_positions = [2]
positions = [word_position, label_position]
positions.extend(other_positions)

def readDataset(windowSize, word2Idx, caseLookup):

    # Read in data
    print "Read in data and create matrices"
    ner_train_sentences = GermEvalReader.readFile(ner_trainFile, 0, 3)
    ner_dev_sentences = GermEvalReader.readFile(ner_devFile, 0, 3)
    ner_test_sentences = GermEvalReader.readFile(ner_testFile, 0, 3)

    #Label mapping for POS
    ner_label2Idx, ner_idx2Label = GermEvalReader.getLabelDict(ner_trainFile)

    # Create numpy arrays
    ner_train_x, ner_train_case_x, ner_train_y = GermEvalReader.createNumpyArrayWithCasing(
        ner_train_sentences, windowSize, word2Idx, ner_label2Idx, caseLookup)
    ner_dev_x, ner_dev_case_x, ner_dev_y = GermEvalReader.createNumpyArrayWithCasing(ner_dev_sentences,
                                                                                                 windowSize, word2Idx,
                                                                                                 ner_label2Idx,
                                                                                                 caseLookup)
    ner_test_x, ner_test_case_x, ner_test_y = GermEvalReader.createNumpyArrayWithCasing(ner_test_sentences,
                                                                                                    windowSize,
                                                                                                    word2Idx,
                                                                                                    ner_label2Idx,
                                                                                                    caseLookup)

    ner_train_y_cat = np_utils.to_categorical(ner_train_y, len(ner_label2Idx))

    return (ner_train_x, ner_train_case_x, ner_train_y, ner_train_y_cat), (ner_dev_x, ner_dev_case_x, ner_dev_y), (
    ner_test_x, ner_test_case_x, ner_test_y)

def readDatasetExt(windowSize, word2Idx, caseLookup):
    # load data
    ner_train_sentences = GermEvalReader.readFileExt(ner_trainFileExt)
    ner_dev_sentences = GermEvalReader.readFileExt(ner_devFileExt)
    ner_test_sentences = GermEvalReader.readFileExt(ner_testFileExt)

    # create dictionaries
    # Label mapping for POS
    label_column_train = filterColumn(ner_train_sentences, label_position)
    word_column_train = filterColumn(ner_train_sentences, word_position)
    pos_column_train = filterColumn(ner_train_sentences, pos_position)

    label_column_dev = filterColumn(ner_dev_sentences, label_position)
    word_column_dev = filterColumn(ner_dev_sentences, word_position)
    pos_column_dev = filterColumn(ner_dev_sentences, pos_position)

    label_column_test = filterColumn(ner_test_sentences, label_position)
    word_column_test = filterColumn(ner_test_sentences, word_position)
    pos_column_test = filterColumn(ner_test_sentences, pos_position)

    ner_label2Idx, ner_idx2Label = DatasetExtender.getDict(label_column_train)
    ner_pos2Idx, ner_idx2pos = DatasetExtender.getDict(pos_column_train, withAddLabels=True)

    # convert value to index
    words_train = GermEvalReader.convertValue2Idx(word_column_train, word2Idx, GermEvalReader.wordConverter)
    pos_train = GermEvalReader.convertValue2Idx(pos_column_train, ner_pos2Idx, GermEvalReader.wordConverter)
    casing_train = GermEvalReader.convertValue2Idx(word_column_train, caseLookup, GermEvalReader.getCasing)
    labels_train = GermEvalReader.convertValue2Idx(label_column_train, ner_label2Idx, GermEvalReader.labelConverter)

    words_dev = GermEvalReader.convertValue2Idx(word_column_dev, word2Idx, GermEvalReader.wordConverter)
    pos_dev = GermEvalReader.convertValue2Idx(pos_column_dev, ner_pos2Idx, GermEvalReader.wordConverter)
    casing_dev = GermEvalReader.convertValue2Idx(word_column_dev, caseLookup, GermEvalReader.getCasing)
    labels_dev = GermEvalReader.convertValue2Idx(label_column_dev, ner_label2Idx, GermEvalReader.labelConverter)

    words_test = GermEvalReader.convertValue2Idx(word_column_test, word2Idx, GermEvalReader.wordConverter)
    pos_test = GermEvalReader.convertValue2Idx(pos_column_test, ner_pos2Idx, GermEvalReader.wordConverter)
    casing_test = GermEvalReader.convertValue2Idx(word_column_test, caseLookup, GermEvalReader.getCasing)
    labels_test = GermEvalReader.convertValue2Idx(label_column_test, ner_label2Idx, GermEvalReader.labelConverter)

    # create numpy datasets
    ner_train_x = GermEvalReader.createNumpyArray(words_train, windowSize, word2Idx)
    ner_train_pos_x = GermEvalReader.createNumpyArray(pos_train, windowSize, ner_pos2Idx)
    ner_train_casing_x = GermEvalReader.createNumpyArray(casing_train, windowSize, caseLookup)
    ner_train_y = np.concatenate(labels_train)

    ner_dev_x = GermEvalReader.createNumpyArray(words_dev, windowSize, word2Idx)
    ner_dev_pos_x = GermEvalReader.createNumpyArray(pos_dev, windowSize, ner_pos2Idx)
    ner_dev_casing_x = GermEvalReader.createNumpyArray(casing_dev, windowSize, caseLookup)
    ner_dev_y = np.concatenate(labels_dev)

    ner_test_x = GermEvalReader.createNumpyArray(words_test, windowSize, word2Idx)
    ner_test_pos_x = GermEvalReader.createNumpyArray(pos_test, windowSize, ner_pos2Idx)
    ner_test_casing_x = GermEvalReader.createNumpyArray(casing_test, windowSize, caseLookup)
    ner_test_y = np.concatenate(labels_test)

    print "shape of ner_train_x:", ner_train_x.shape
    print ner_train_x[0]

    print "shape of ner_train_pos_x:", ner_train_pos_x.shape
    print ner_train_pos_x[0]

    print "shape of ner_train_casing_x:", ner_train_casing_x.shape
    print ner_train_casing_x[0]

    print "shape of ner_train_y:", ner_train_y.shape
    print ner_train_y



    print "shape of ner_dev_x:", ner_dev_x.shape
    print ner_dev_x[0]

    print "shape of ner_dev_pos_x:", ner_dev_pos_x.shape
    print ner_dev_pos_x[0]

    print "shape of ner_dev_casing_x:", ner_dev_casing_x.shape
    print ner_dev_casing_x[0]

    print "shape of ner_dev_y:", ner_dev_y.shape
    print ner_dev_y



    print "shape of ner_test_x:", ner_test_x.shape
    print ner_test_x[0]

    print "shape of ner_test_pos_x:", ner_test_pos_x.shape
    print ner_test_pos_x[0]

    print "shape of ner_test_casing_x:", ner_test_casing_x.shape
    print ner_test_casing_x[0]

    print "shape of ner_test_y:", ner_test_y.shape
    print ner_test_y



    input_train = [ner_train_x, ner_train_pos_x, ner_train_casing_x]
    input_dev = [ner_dev_x, ner_dev_pos_x, ner_dev_casing_x]
    input_test = [ner_test_x, ner_test_pos_x, ner_test_casing_x]

    ner_train_y_cat = np_utils.to_categorical(ner_train_y, len(ner_label2Idx))

    dicts = [word2Idx, ner_pos2Idx, caseLookup]
    return [input_train, ner_train_y_cat], [input_dev, ner_dev_y], [input_test, ner_test_y], dicts

def filterColumn(sentences, position):
    return map(lambda sentence: sentence[:, position], sentences)

def extendDataset(filename, train_extensions, dev_extensions, test_extensions):
    ner_train_sentences = GermEvalReader.readFile(ner_trainFile, 0, 3)
    ner_dev_sentences = GermEvalReader.readFile(ner_devFile, 0, 3)
    ner_test_sentences = GermEvalReader.readFile(ner_testFile, 0, 3)

    filename, file_extension = path.splitext(filename)

    DatasetExtender.extendDataset("{0}_train_ext{1}".format(filename, file_extension), ner_train_sentences, train_extensions)
    DatasetExtender.extendDataset("{0}_dev_ext{1}".format(filename, file_extension), ner_dev_sentences, dev_extensions)
    DatasetExtender.extendDataset("{0}_test_ext{1}".format(filename, file_extension), ner_test_sentences, test_extensions)

def getLabelDict():
    return GermEvalReader.getLabelDict(ner_trainFile)