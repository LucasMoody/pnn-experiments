from datasets import GermEvalReader, DatasetExtender
from keras.utils import np_utils
from os import path
import numpy as np

events_trainFile = 'datasets/ace_ed/data/train.txt'
events_devFile = 'datasets/ace_ed/data/dev.txt'
events_testFile = 'datasets/ace_ed/data/test.txt'

trainFileExt = 'datasets/ace_ed/data/events_train_ext.conllu'
devFileExt = 'datasets/ace_ed/data/events_dev_ext.conllu'
testFileExt = 'datasets/ace_ed/data/events_test_ext.conllu'

word_position = 0
label_position = 4

ext_word_position = 0
ext_label_position = 1
ext_pos_position = 2
ext_ner_position = 3
ext_ecb_ed_position = 4
ext_tac_ed_position = 5
ext_tempeval_ed_position = 6

def readDataset(windowSize, word2Idx, case2Idx):

    # Read in data
    print "Read in data and create matrices"
    events_train_sentences = GermEvalReader.readFile(events_trainFile, word_position, label_position)
    events_dev_sentences = GermEvalReader.readFile(events_devFile, word_position, label_position)
    events_test_sentences = GermEvalReader.readFile(events_testFile, word_position, label_position)

    #Label mapping for ED
    events_label2Idx, events_idx2Label = GermEvalReader.getLabelDict(events_trainFile, label_position)

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
    word_column_train = filterColumn(train_sentences, word_position)
    word_column_dev = filterColumn(dev_sentences, word_position)
    word_column_test = filterColumn(test_sentences, word_position)

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
    label_column_train = filterColumn(train_sentences, label_position)
    label_column_dev = filterColumn(dev_sentences, label_position)
    label_column_test = filterColumn(test_sentences, label_position)

    # create dictionaries
    events_label2Idx, events_idx2Label = DatasetExtender.getDict(label_column_train)

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
    pos_column_train = filterColumn(train_sentences, ext_pos_position)
    pos_column_dev = filterColumn(dev_sentences, ext_pos_position)
    pos_column_test = filterColumn(test_sentences, ext_pos_position)

    events_pos2Idx, events_idx2pos = DatasetExtender.getDict(pos_column_train, withAddLabels=True)

    pos_train = GermEvalReader.convertValue2Idx(pos_column_train, events_pos2Idx, GermEvalReader.wordConverter)
    pos_dev = GermEvalReader.convertValue2Idx(pos_column_dev, events_pos2Idx, GermEvalReader.wordConverter)
    pos_test = GermEvalReader.convertValue2Idx(pos_column_test, events_pos2Idx, GermEvalReader.wordConverter)

    events_train_pos_x = GermEvalReader.createNumpyArray(pos_train, windowSize, events_pos2Idx)
    events_dev_pos_x = GermEvalReader.createNumpyArray(pos_dev, windowSize, events_pos2Idx)
    events_test_pos_x = GermEvalReader.createNumpyArray(pos_test, windowSize, events_pos2Idx)

    # ------ NER ------ #

    ner_column_train = filterColumn(train_sentences, ext_ner_position)
    ner_column_dev = filterColumn(dev_sentences, ext_ner_position)
    ner_column_test = filterColumn(test_sentences, ext_ner_position)

    events_ner2Idx, events_ner2pos = DatasetExtender.getDict(ner_column_train, withAddLabels=True)

    ner_train = GermEvalReader.convertValue2Idx(ner_column_train, events_ner2Idx, GermEvalReader.wordConverter)
    ner_dev = GermEvalReader.convertValue2Idx(ner_column_dev, events_ner2Idx, GermEvalReader.wordConverter)
    ner_test = GermEvalReader.convertValue2Idx(ner_column_test, events_ner2Idx, GermEvalReader.wordConverter)

    events_train_ner_x = GermEvalReader.createNumpyArray(ner_train, windowSize, events_ner2Idx)
    events_dev_ner_x = GermEvalReader.createNumpyArray(ner_dev, windowSize, events_ner2Idx)
    events_test_ner_x = GermEvalReader.createNumpyArray(ner_test, windowSize, events_ner2Idx)
    
    # ----- ECB PLUS ED ----- #

    ecb_ed_column_train = filterColumn(train_sentences, ext_ecb_ed_position)
    ecb_ed_column_dev = filterColumn(dev_sentences, ext_ecb_ed_position)
    ecb_ed_column_test = filterColumn(test_sentences, ext_ecb_ed_position)

    events_ecb_ed2Idx, events_ecb_ed2pos = DatasetExtender.getDict(ecb_ed_column_train, withAddLabels=True)

    ecb_ed_train = GermEvalReader.convertValue2Idx(ecb_ed_column_train, events_ecb_ed2Idx, GermEvalReader.wordConverter)
    ecb_ed_dev = GermEvalReader.convertValue2Idx(ecb_ed_column_dev, events_ecb_ed2Idx, GermEvalReader.wordConverter)
    ecb_ed_test = GermEvalReader.convertValue2Idx(ecb_ed_column_test, events_ecb_ed2Idx, GermEvalReader.wordConverter)

    events_train_ecb_ed_x = GermEvalReader.createNumpyArray(ecb_ed_train, windowSize, events_ecb_ed2Idx)
    events_dev_ecb_ed_x = GermEvalReader.createNumpyArray(ecb_ed_dev, windowSize, events_ecb_ed2Idx)
    events_test_ecb_ed_x = GermEvalReader.createNumpyArray(ecb_ed_test, windowSize, events_ecb_ed2Idx)
    
    # ----- TAC 2015 ED ----- #

    tac_ed_column_train = filterColumn(train_sentences, ext_tac_ed_position)
    tac_ed_column_dev = filterColumn(dev_sentences, ext_tac_ed_position)
    tac_ed_column_test = filterColumn(test_sentences, ext_tac_ed_position)

    events_tac_ed2Idx, events_tac_ed2pos = DatasetExtender.getDict(tac_ed_column_train, withAddLabels=True)

    tac_ed_train = GermEvalReader.convertValue2Idx(tac_ed_column_train, events_tac_ed2Idx, GermEvalReader.wordConverter)
    tac_ed_dev = GermEvalReader.convertValue2Idx(tac_ed_column_dev, events_tac_ed2Idx, GermEvalReader.wordConverter)
    tac_ed_test = GermEvalReader.convertValue2Idx(tac_ed_column_test, events_tac_ed2Idx, GermEvalReader.wordConverter)

    events_train_tac_ed_x = GermEvalReader.createNumpyArray(tac_ed_train, windowSize, events_tac_ed2Idx)
    events_dev_tac_ed_x = GermEvalReader.createNumpyArray(tac_ed_dev, windowSize, events_tac_ed2Idx)
    events_test_tac_ed_x = GermEvalReader.createNumpyArray(tac_ed_test, windowSize, events_tac_ed2Idx)
    
    # ----- TEMPEVAL 3 ED ----- #

    tempeval_ed_column_train = filterColumn(train_sentences, ext_tempeval_ed_position)
    tempeval_ed_column_dev = filterColumn(dev_sentences, ext_tempeval_ed_position)
    tempeval_ed_column_test = filterColumn(test_sentences, ext_tempeval_ed_position)

    events_tempeval_ed2Idx, events_tempeval_ed2pos = DatasetExtender.getDict(tempeval_ed_column_train, withAddLabels=True)

    tempeval_ed_train = GermEvalReader.convertValue2Idx(tempeval_ed_column_train, events_tempeval_ed2Idx, GermEvalReader.wordConverter)
    tempeval_ed_dev = GermEvalReader.convertValue2Idx(tempeval_ed_column_dev, events_tempeval_ed2Idx, GermEvalReader.wordConverter)
    tempeval_ed_test = GermEvalReader.convertValue2Idx(tempeval_ed_column_test, events_tempeval_ed2Idx, GermEvalReader.wordConverter)

    events_train_tempeval_ed_x = GermEvalReader.createNumpyArray(tempeval_ed_train, windowSize, events_tempeval_ed2Idx)
    events_dev_tempeval_ed_x = GermEvalReader.createNumpyArray(tempeval_ed_dev, windowSize, events_tempeval_ed2Idx)
    events_test_tempeval_ed_x = GermEvalReader.createNumpyArray(tempeval_ed_test, windowSize, events_tempeval_ed2Idx)

    # ----- PREPARE RESULT ----- #
    input_train = [events_train_x, events_train_casing_x, events_train_pos_x, events_train_ner_x, events_train_ecb_ed_x, events_train_tac_ed_x, events_train_tempeval_ed_x]
    input_dev = [events_dev_x, events_dev_casing_x, events_dev_pos_x, events_dev_ner_x, events_dev_ecb_ed_x, events_dev_tac_ed_x, events_dev_tempeval_ed_x]
    input_test = [events_test_x, events_test_casing_x, events_test_pos_x, events_test_ner_x, events_test_ecb_ed_x, events_test_tac_ed_x, events_test_tempeval_ed_x]

    events_train_y_cat = np_utils.to_categorical(events_train_y, len(events_label2Idx))

    dicts = [word2Idx, events_pos2Idx, events_ner2Idx, events_ecb_ed2Idx, events_tac_ed2Idx, events_tempeval_ed2Idx, case2Idx, events_label2Idx, events_idx2Label]
    return [input_train, events_train_y_cat], [input_dev, events_dev_y], [input_test, events_test_y], dicts

def filterColumn(sentences, position):
    return map(lambda sentence: sentence[:, position], sentences)

def extendDataset(filename, train_extensions, dev_extensions, test_extensions):
    train_sentences = GermEvalReader.readFile(events_trainFile, word_position, label_position)
    dev_sentences = GermEvalReader.readFile(events_devFile, word_position, label_position)
    test_sentences = GermEvalReader.readFile(events_testFile, word_position, label_position)

    filename, file_extension = path.splitext(filename)

    DatasetExtender.extendDataset("{0}_train_ext{1}".format(filename, file_extension), train_sentences, train_extensions)
    DatasetExtender.extendDataset("{0}_dev_ext{1}".format(filename, file_extension), dev_sentences, dev_extensions)
    DatasetExtender.extendDataset("{0}_test_ext{1}".format(filename, file_extension), test_sentences, test_extensions)

def getLabelDict():
    return GermEvalReader.getLabelDict(events_trainFile, label_position)