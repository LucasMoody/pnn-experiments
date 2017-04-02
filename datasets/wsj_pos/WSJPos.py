from datasets import GermEvalReader, DatasetExtender
from keras.utils import np_utils
from os import path
import numpy as np

pos_trainFile = 'datasets/wsj_pos/data/train.txt'
pos_devFile = 'datasets/wsj_pos/data/dev.txt'
pos_testFile = 'datasets/wsj_pos/data/test.txt'

pos_trainFileExt = 'datasets/wsj_pos/data/wsj_train_ext.conllu'
pos_devFileExt = 'datasets/wsj_pos/data/wsj_dev_ext.conllu'
pos_testFileExt = 'datasets/wsj_pos/data/wsj_test_ext.conllu'

word_position = 0
label_position = 1
ner_position = 2
chunking_position = 3
ud_pos_position = 4
other_positions = [ner_position, chunking_position, ud_pos_position]
positions = [word_position, label_position]
positions.extend(other_positions)


def readDataset(windowSize, word2Idx, caseLookup):
    # Read in data
    print "Read in data and create matrices"
    pos_train_sentences = GermEvalReader.readFile(pos_trainFile, 0, 1)
    pos_dev_sentences = GermEvalReader.readFile(pos_devFile, 0, 1)
    pos_test_sentences = GermEvalReader.readFile(pos_testFile, 0, 1)

    # Label mapping for POS
    pos_label2Idx, pos_idx2Label = GermEvalReader.getLabelDict(pos_trainFile, tagPosition=1)

    # Create numpy arrays
    pos_train_x, pos_train_case_x, pos_train_y = GermEvalReader.createNumpyArrayWithCasing(pos_train_sentences,
                                                                                           windowSize, word2Idx,
                                                                                           pos_label2Idx, caseLookup)
    pos_dev_x, pos_dev_case_x, pos_dev_y = GermEvalReader.createNumpyArrayWithCasing(pos_dev_sentences, windowSize,
                                                                                     word2Idx, pos_label2Idx,
                                                                                     caseLookup)
    pos_test_x, pos_test_case_x, pos_test_y = GermEvalReader.createNumpyArrayWithCasing(pos_test_sentences, windowSize,
                                                                                        word2Idx, pos_label2Idx,
                                                                                        caseLookup)

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
    label_column_train = filterColumn(pos_train_sentences, label_position)
    word_column_train = filterColumn(pos_train_sentences, word_position)
    ner_column_train = filterColumn(pos_train_sentences, ner_position)
    chunking_column_train = filterColumn(pos_train_sentences, chunking_position)
    ud_pos_column_train = filterColumn(pos_train_sentences, ud_pos_position)

    label_column_dev = filterColumn(pos_dev_sentences, label_position)
    word_column_dev = filterColumn(pos_dev_sentences, word_position)
    ner_column_dev = filterColumn(pos_dev_sentences, ner_position)
    chunking_column_dev = filterColumn(pos_dev_sentences, chunking_position)
    ud_pos_column_dev = filterColumn(pos_dev_sentences, ud_pos_position)

    label_column_test = filterColumn(pos_test_sentences, label_position)
    word_column_test = filterColumn(pos_test_sentences, word_position)
    ner_column_test = filterColumn(pos_test_sentences, ner_position)
    chunking_column_test = filterColumn(pos_test_sentences, chunking_position)
    ud_pos_column_test = filterColumn(pos_test_sentences, ud_pos_position)

    pos_label2Idx, pos_idx2Label = DatasetExtender.getDict(label_column_train)
    pos_ner2Idx, pos_idx2ner = DatasetExtender.getDict(ner_column_train, withAddLabels=True)
    pos_chunking2Idx, pos_idx2chunking = DatasetExtender.getDict(chunking_column_train, withAddLabels=True)
    pos_ud_pos2Idx, pos_idx2ud_pos = DatasetExtender.getDict(ud_pos_column_train, withAddLabels=True)

    # convert value to index
    words_train = GermEvalReader.convertValue2Idx(word_column_train, word2Idx, GermEvalReader.wordConverter)
    ner_train = GermEvalReader.convertValue2Idx(ner_column_train, pos_ner2Idx, GermEvalReader.wordConverter)
    chunking_train = GermEvalReader.convertValue2Idx(chunking_column_train, pos_chunking2Idx,
                                                     GermEvalReader.wordConverter)
    ud_pos_train = GermEvalReader.convertValue2Idx(ud_pos_column_train, pos_ud_pos2Idx, GermEvalReader.wordConverter)
    casing_train = GermEvalReader.convertValue2Idx(word_column_train, case2Idx, GermEvalReader.getCasing)
    labels_train = GermEvalReader.convertValue2Idx(label_column_train, pos_label2Idx, GermEvalReader.labelConverter)

    words_dev = GermEvalReader.convertValue2Idx(word_column_dev, word2Idx, GermEvalReader.wordConverter)
    ner_dev = GermEvalReader.convertValue2Idx(ner_column_dev, pos_ner2Idx, GermEvalReader.wordConverter)
    chunking_dev = GermEvalReader.convertValue2Idx(chunking_column_dev, pos_chunking2Idx, GermEvalReader.wordConverter)
    ud_pos_dev = GermEvalReader.convertValue2Idx(ud_pos_column_dev, pos_ud_pos2Idx, GermEvalReader.wordConverter)
    casing_dev = GermEvalReader.convertValue2Idx(word_column_dev, case2Idx, GermEvalReader.getCasing)
    labels_dev = GermEvalReader.convertValue2Idx(label_column_dev, pos_label2Idx, GermEvalReader.labelConverter)

    words_test = GermEvalReader.convertValue2Idx(word_column_test, word2Idx, GermEvalReader.wordConverter)
    ner_test = GermEvalReader.convertValue2Idx(ner_column_test, pos_ner2Idx, GermEvalReader.wordConverter)
    chunking_test = GermEvalReader.convertValue2Idx(chunking_column_test, pos_chunking2Idx,
                                                    GermEvalReader.wordConverter)
    ud_pos_test = GermEvalReader.convertValue2Idx(ud_pos_column_test, pos_ud_pos2Idx, GermEvalReader.wordConverter)
    casing_test = GermEvalReader.convertValue2Idx(word_column_test, case2Idx, GermEvalReader.getCasing)
    labels_test = GermEvalReader.convertValue2Idx(label_column_test, pos_label2Idx, GermEvalReader.labelConverter)

    # create numpy datasets
    pos_train_x = GermEvalReader.createNumpyArray(words_train, windowSize, word2Idx)
    pos_train_ner_x = GermEvalReader.createNumpyArray(ner_train, windowSize, pos_ner2Idx)
    pos_train_chunking_x = GermEvalReader.createNumpyArray(chunking_train, windowSize, pos_chunking2Idx)
    pos_train_ud_pos_x = GermEvalReader.createNumpyArray(ud_pos_train, windowSize, pos_ud_pos2Idx)
    pos_train_casing_x = GermEvalReader.createNumpyArray(casing_train, windowSize, case2Idx)
    pos_train_y = np.concatenate(labels_train)

    pos_dev_x = GermEvalReader.createNumpyArray(words_dev, windowSize, word2Idx)
    pos_dev_ner_x = GermEvalReader.createNumpyArray(ner_dev, windowSize, pos_ner2Idx)
    pos_dev_chunking_x = GermEvalReader.createNumpyArray(chunking_dev, windowSize, pos_chunking2Idx)
    pos_dev_ud_pos_x = GermEvalReader.createNumpyArray(ud_pos_dev, windowSize, pos_ud_pos2Idx)
    pos_dev_casing_x = GermEvalReader.createNumpyArray(casing_dev, windowSize, case2Idx)
    pos_dev_y = np.concatenate(labels_dev)

    pos_test_x = GermEvalReader.createNumpyArray(words_test, windowSize, word2Idx)
    pos_test_ner_x = GermEvalReader.createNumpyArray(ner_test, windowSize, pos_ner2Idx)
    pos_test_chunking_x = GermEvalReader.createNumpyArray(chunking_test, windowSize, pos_chunking2Idx)
    pos_test_ud_pos_x = GermEvalReader.createNumpyArray(ud_pos_test, windowSize, pos_ud_pos2Idx)
    pos_test_casing_x = GermEvalReader.createNumpyArray(casing_test, windowSize, case2Idx)
    pos_test_y = np.concatenate(labels_test)

    print "shape of pos_train_x:", pos_train_x.shape
    print pos_train_x[0]

    print "shape of pos_train_ner_x:", pos_train_ner_x.shape
    print pos_train_ner_x[0]

    print "shape of pos_train_chunking_x:", pos_train_chunking_x.shape
    print pos_train_chunking_x[0]

    print "shape of pos_train_ud_pos_x:", pos_train_ud_pos_x.shape
    print pos_train_ud_pos_x[0]

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

    print "shape of pos_dev_ud_pos_x:", pos_dev_ud_pos_x.shape
    print pos_dev_ud_pos_x[0]

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

    print "shape of pos_test_ud_pos_x:", pos_test_ud_pos_x.shape
    print pos_test_ud_pos_x[0]

    print "shape of pos_test_casing_x:", pos_test_casing_x.shape
    print pos_test_casing_x[0]

    print "shape of pos_test_y:", pos_test_y.shape
    print pos_test_y

    input_train = [pos_train_x, pos_train_ner_x, pos_train_chunking_x, pos_train_ud_pos_x, pos_train_casing_x]
    input_dev = [pos_dev_x, pos_dev_ner_x, pos_dev_chunking_x, pos_dev_ud_pos_x, pos_dev_casing_x]
    input_test = [pos_test_x, pos_test_ner_x, pos_test_chunking_x, pos_test_ud_pos_x, pos_test_casing_x]

    pos_train_y_cat = np_utils.to_categorical(pos_train_y, len(pos_label2Idx))

    dicts = [word2Idx, pos_ner2Idx, pos_chunking2Idx, pos_ud_pos2Idx, case2Idx, pos_label2Idx, pos_idx2Label]
    return [input_train, pos_train_y_cat], [input_dev, pos_dev_y], [input_test, pos_test_y], dicts


def filterColumn(sentences, position):
    return map(lambda sentence: sentence[:, position], sentences)


def extendDataset(filename, train_extensions, dev_extensions, test_extensions):
    pos_train_sentences = GermEvalReader.readFile(pos_trainFile, 0, 1)
    pos_dev_sentences = GermEvalReader.readFile(pos_devFile, 0, 1)
    pos_test_sentences = GermEvalReader.readFile(pos_testFile, 0, 1)

    filename, file_extension = path.splitext(filename)

    DatasetExtender.extendDataset("{0}_train_ext{1}".format(filename, file_extension), pos_train_sentences,
                                  train_extensions)
    DatasetExtender.extendDataset("{0}_dev_ext{1}".format(filename, file_extension), pos_dev_sentences, dev_extensions)
    DatasetExtender.extendDataset("{0}_test_ext{1}".format(filename, file_extension), pos_test_sentences,
                                  test_extensions)


def getLabelDict():
    return GermEvalReader.getLabelDict(pos_trainFile, tagPosition=1)
