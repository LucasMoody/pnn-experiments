from datasets import GermEvalReader, DatasetExtender
from keras.utils import np_utils
from os import path
import numpy as np

events_trainFile = 'datasets/tempeval3_ed/data/train.txt'
events_devFile = 'datasets/tempeval3_ed/data/dev.txt'
events_testFile = 'datasets/tempeval3_ed/data/test.txt'

trainFileExt = 'datasets/tempeval3_ed/data/train_ext.conllu'
devFileExt = 'datasets/tempeval3_ed/data/dev_ext.conllu'
testFileExt = 'datasets/tempeval3_ed/data/test_ext.conllu'

directory = 'datasets/tempeval3_ed/data/'

word_position = 0
label_position = 2

ext_word_position = 0
ext_label_position = 1
ext_pos_position = 2
ext_ner_position = 3
ext_chunking_position = 4
ext_ace_position = 5
ext_ecb_position = 6
ext_tac_position = 7

def readDataset(windowSize, word2Idx, case2Idx):

    # Read in data
    print "Read in data and create matrices"
    events_train_sentences = GermEvalReader.readFile(events_trainFile, word_position, label_position)
    events_dev_sentences = GermEvalReader.readFile(events_devFile, word_position, label_position)
    events_test_sentences = GermEvalReader.readFile(events_testFile, word_position, label_position)

    #Label mapping for ED
    events_label2Idx, events_idx2Label = GermEvalReader.getLabelDict(events_trainFile, label_position)

    # there is a tag in the test file which does not appear in the train file
    # so the dictionaries have to be updated in order not to get an error
    dev_label_dicts = GermEvalReader.getLabelDict(events_devFile, label_position)
    test_label_dicts = GermEvalReader.getLabelDict(events_testFile, label_position)
    for tag in dev_label_dicts[0]:
        if tag not in events_label2Idx:
            events_label2Idx[tag] = len(events_label2Idx)
    for tag in test_label_dicts[0]:
        if tag not in events_label2Idx:
            events_label2Idx[tag] = len(events_label2Idx)
    events_idx2Label = {v: k for k, v in events_label2Idx.items()}

    # Create numpy arrays
    events_train_x, events_train_case_x, events_train_y = GermEvalReader.createNumpyArrayWithCasing(
        events_train_sentences, windowSize, word2Idx, events_label2Idx, case2Idx)
    events_dev_x, events_dev_case_x, events_dev_y = GermEvalReader.createNumpyArrayWithCasing(events_dev_sentences,
                                                                                     windowSize, word2Idx,
                                                                                     events_label2Idx,
                                                                                     case2Idx)
    events_test_x, events_test_case_x, events_test_y = GermEvalReader.createNumpyArrayWithCasing(events_test_sentences,
                                                                                        windowSize,
                                                                                        word2Idx,
                                                                                        events_label2Idx,
                                                                                        case2Idx)
    events_input_train = [events_train_x, events_train_case_x]
    events_input_dev = [events_dev_x, events_dev_case_x]
    events_input_test = [events_test_x, events_test_case_x]

    events_train_y_cat = np_utils.to_categorical(events_train_y, len(events_label2Idx))

    events_dicts = [word2Idx, case2Idx, events_label2Idx, events_idx2Label]
    return [events_input_train, events_train_y_cat], [events_input_dev, events_dev_y], [events_input_test, events_test_y], events_dicts

def readDatasetExt(windowSize, word2Idx, case2Idx):
    # load data
    train_sentences = GermEvalReader.readFileExt(trainFileExt)
    dev_sentences = GermEvalReader.readFileExt(devFileExt)
    test_sentences = GermEvalReader.readFileExt(testFileExt)

    # ----- WORDS ----- #
    # get words from sentences
    word_column_train = DatasetExtender.filterColumn(train_sentences, ext_word_position)
    word_column_dev = DatasetExtender.filterColumn(dev_sentences, ext_word_position)
    word_column_test = DatasetExtender.filterColumn(test_sentences, ext_word_position)

    # convert them to an index of the word embedding
    words_train = GermEvalReader.convertValue2Idx(word_column_train, word2Idx, GermEvalReader.wordConverter)
    words_dev = GermEvalReader.convertValue2Idx(word_column_dev, word2Idx, GermEvalReader.wordConverter)
    words_test = GermEvalReader.convertValue2Idx(word_column_test, word2Idx, GermEvalReader.wordConverter)

    # convert them into numbers according the window
    events_train_x = GermEvalReader.createNumpyArray(words_train, windowSize, word2Idx)
    events_dev_x = GermEvalReader.createNumpyArray(words_dev, windowSize, word2Idx)
    events_test_x = GermEvalReader.createNumpyArray(words_test, windowSize, word2Idx)

    # ----- LABELS ----- #

    # get labels from sentences
    label_column_train = DatasetExtender.filterColumn(train_sentences, ext_label_position)
    label_column_dev = DatasetExtender.filterColumn(dev_sentences, ext_label_position)
    label_column_test = DatasetExtender.filterColumn(test_sentences, ext_label_position)

    # create dictionaries
    events_label2Idx, events_idx2Label = DatasetExtender.getDict(label_column_train)
    # there is a tag in the test file which does not appear in the train file
    # so the dictionaries have to be updated in order not to get an error
    dev_label_dicts = DatasetExtender.getDict(label_column_dev)
    test_label_dicts = DatasetExtender.getDict(label_column_test)
    for tag in dev_label_dicts[0]:
        if tag not in events_label2Idx:
            events_label2Idx[tag] = len(events_label2Idx)
    for tag in test_label_dicts[0]:
        if tag not in events_label2Idx:
            events_label2Idx[tag] = len(events_label2Idx)
    events_idx2Label = {v: k for k, v in events_label2Idx.items()}

    # convert labels into index
    labels_train = GermEvalReader.convertValue2Idx(label_column_train, events_label2Idx, GermEvalReader.labelConverter)
    labels_dev = GermEvalReader.convertValue2Idx(label_column_dev, events_label2Idx, GermEvalReader.labelConverter)
    labels_test = GermEvalReader.convertValue2Idx(label_column_test, events_label2Idx, GermEvalReader.labelConverter)

    # concatenate them to final label set
    events_train_y = np.concatenate(labels_train)
    events_dev_y = np.concatenate(labels_dev)
    events_test_y = np.concatenate(labels_test)

    # ----- CASING ----- #

    # convert words into casing
    casing_train = GermEvalReader.convertValue2Idx(word_column_train, case2Idx, GermEvalReader.getCasing)
    casing_dev = GermEvalReader.convertValue2Idx(word_column_dev, case2Idx, GermEvalReader.getCasing)
    casing_test = GermEvalReader.convertValue2Idx(word_column_test, case2Idx, GermEvalReader.getCasing)

    # convert them into numbers according the window
    events_train_casing_x = GermEvalReader.createNumpyArray(casing_train, windowSize, case2Idx)
    events_dev_casing_x = GermEvalReader.createNumpyArray(casing_dev, windowSize, case2Idx)
    events_test_casing_x = GermEvalReader.createNumpyArray(casing_test, windowSize, case2Idx)

    # ----- POS ----- #
    events_train_pos_x, events_dev_pos_x, events_test_pos_x, events_pos2Idx = DatasetExtender.createNumpyArraysForFeature(
        train_sentences, dev_sentences, test_sentences, ext_pos_position, GermEvalReader.wordConverter, windowSize)

    # ------ NER ------ #
    events_train_ner_x, events_dev_ner_x, events_test_ner_x, events_ner2Idx = DatasetExtender.createNumpyArraysForFeature(
        train_sentences,
        dev_sentences, test_sentences,
        ext_ner_position,
        GermEvalReader.wordConverter,
        windowSize)
    # ------ CHUNKING ------ #
    events_train_chunking_x, events_dev_chunking_x, events_test_chunking_x, events_chunking2Idx = DatasetExtender.createNumpyArraysForFeature(
        train_sentences,
        dev_sentences, test_sentences,
        ext_chunking_position,
        GermEvalReader.wordConverter,
        windowSize)

    # ----- ACE ED ----- #
    events_train_ace_x, events_dev_ace_x, events_test_ace_x, events_ace2Idx = DatasetExtender.createNumpyArraysForFeature(
        train_sentences,
        dev_sentences, test_sentences,
        ext_ace_position,
        GermEvalReader.wordConverter,
        windowSize)

    # ----- ECB+ ED ----- #
    events_train_ecb_x, events_dev_ecb_x, events_test_ecb_x, events_ecb2Idx = DatasetExtender.createNumpyArraysForFeature(
        train_sentences,
        dev_sentences, test_sentences,
        ext_ecb_position,
        GermEvalReader.wordConverter,
        windowSize)

    # ----- TAC 2015 ED ----- #
    events_train_tac_x, events_dev_tac_x, events_test_tac_x, events_tac2Idx = DatasetExtender.createNumpyArraysForFeature(
        train_sentences,
        dev_sentences, test_sentences,
        ext_tac_position,
        GermEvalReader.wordConverter,
        windowSize)

    # PRINT SOME SANITY CHECKS
    print 'Words - No: {0}/{1}/{2}, labelno {3}'.format(events_train_x.shape[0], events_dev_x.shape[0],
                                                        events_test_x.shape[0], len(word2Idx))
    print 'Casing - No: {0}/{1}/{2}, labelno {3}'.format(events_train_casing_x.shape[0], events_dev_casing_x.shape[0],
                                                         events_test_casing_x.shape[0], len(case2Idx))
    print 'POS - No: {0}/{1}/{2}, labelno {3}'.format(events_train_pos_x.shape[0], events_dev_pos_x.shape[0],
                                                      events_test_pos_x.shape[0], len(events_pos2Idx))
    print 'NER - No: {0}/{1}/{2}, labelno {3}'.format(events_train_ner_x.shape[0], events_dev_ner_x.shape[0],
                                                      events_test_ner_x.shape[0], len(events_ner2Idx))
    print 'Chunking - No: {0}/{1}/{2}, labelno {3}'.format(events_train_chunking_x.shape[0],
                                                           events_dev_chunking_x.shape[0],
                                                           events_test_chunking_x.shape[0], len(events_chunking2Idx))
    print 'ACE - No: {0}/{1}/{2}, labelno {3}'.format(events_train_ace_x.shape[0], events_dev_ace_x.shape[0],
                                                       events_test_ace_x.shape[0], len(events_ace2Idx))
    print 'ECB Plus - No: {0}/{1}/{2}, labelno {3}'.format(events_train_ecb_x.shape[0], events_dev_ecb_x.shape[0],
                                                       events_test_ecb_x.shape[0], len(events_ecb2Idx))
    print 'TAC 2015 - No: {0}/{1}/{2}, labelno {3}'.format(events_train_tac_x.shape[0],
                                                           events_dev_tac_x.shape[0],
                                                           events_test_tac_x.shape[0], len(events_tac2Idx))

    # ----- PREPARE RESULT ----- #
    input_train = [events_train_x, events_train_casing_x, events_train_pos_x, events_train_ner_x,
                   events_train_chunking_x, events_train_ace_x, events_train_ecb_x, events_train_tac_x]
    input_dev = [events_dev_x, events_dev_casing_x, events_dev_pos_x, events_dev_ner_x, events_dev_chunking_x,
                 events_dev_ace_x, events_dev_ecb_x, events_dev_tac_x]
    input_test = [events_test_x, events_test_casing_x, events_test_pos_x, events_test_ner_x, events_test_chunking_x,
                  events_test_ace_x, events_test_ecb_x, events_test_tac_x]

    events_train_y_cat = np_utils.to_categorical(events_train_y, len(events_label2Idx))

    dicts = [word2Idx, events_pos2Idx, events_ner2Idx, events_chunking2Idx, events_ace2Idx, events_ecb2Idx,
             events_tac2Idx, case2Idx, events_label2Idx, events_idx2Label]
    return [input_train, events_train_y_cat], [input_dev, events_dev_y], [input_test, events_test_y], dicts

def filterColumn(sentences, position):
    return map(lambda sentence: sentence[:, position], sentences)

def extendDataset(train_extensions, dev_extensions, test_extensions):
    train_sentences = GermEvalReader.readFile(events_trainFile, word_position, label_position)
    dev_sentences = GermEvalReader.readFile(events_devFile, word_position, label_position)
    test_sentences = GermEvalReader.readFile(events_testFile, word_position, label_position)

    DatasetExtender.extendDataset("{0}train_ext.conllu".format(directory), train_sentences, train_extensions)
    DatasetExtender.extendDataset("{0}dev_ext.conllu".format(directory), dev_sentences, dev_extensions)
    DatasetExtender.extendDataset("{0}test_ext.conllu".format(directory), test_sentences, test_extensions)

def getLabelDict():
    events_label2Idx, events_idx2Label = GermEvalReader.getLabelDict(events_trainFile, label_position)
    # there is a tag in the test file which does not appear in the train file
    # so the dictionaries have to be updated in order not to get an error
    dev_label_dicts = GermEvalReader.getLabelDict(events_devFile, label_position)
    test_label_dicts = GermEvalReader.getLabelDict(events_testFile, label_position)
    for tag in dev_label_dicts[0]:
        if tag not in events_label2Idx:
            events_label2Idx[tag] = len(events_label2Idx)
    for tag in test_label_dicts[0]:
        if tag not in events_label2Idx:
            events_label2Idx[tag] = len(events_label2Idx)
    events_idx2Label = {v: k for k, v in events_label2Idx.items()}
    return events_label2Idx, events_idx2Label

def getRawSentences():
    train_sentences = GermEvalReader.readFile(events_trainFile, word_position, label_position)
    dev_sentences = GermEvalReader.readFile(events_devFile, word_position, label_position)
    test_sentences = GermEvalReader.readFile(events_testFile, word_position, label_position)
    return train_sentences, dev_sentences, test_sentences