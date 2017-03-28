from datasets import GermEvalReader, DatasetExtender
from keras.utils import np_utils
from os import path
import numpy as np

events_trainFile = 'datasets/ecbplus_ed/data/train.txt'
events_devFile = 'datasets/ecbplus_ed/data/dev.txt'
events_testFile = 'datasets/ecbplus_ed/data/test.txt'

trainFileExt = 'datasets/ecbplus_ed/data/events_train_ext.conllu'
devFileExt = 'datasets/ecbplus_ed/data/events_dev_ext.conllu'
testFileExt = 'datasets/ecbplus_ed/data/events_test_ext.conllu'

word_position = 0
label_position = 1
pos_position = 2
ner_position = 3
other_positions = [pos_position, ner_position]
positions = [word_position, label_position]
positions.extend(other_positions)

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

    # create dictionaries
    # Label mapping for POS
    label_column_train = filterColumn(train_sentences, label_position)
    word_column_train = filterColumn(train_sentences, word_position)
    pos_column_train = filterColumn(train_sentences, pos_position)
    ner_column_train = filterColumn(train_sentences, ner_position)

    label_column_dev = filterColumn(dev_sentences, label_position)
    word_column_dev = filterColumn(dev_sentences, word_position)
    pos_column_dev = filterColumn(dev_sentences, pos_position)
    ner_column_dev = filterColumn(dev_sentences, ner_position)

    label_column_test = filterColumn(test_sentences, label_position)
    word_column_test = filterColumn(test_sentences, word_position)
    pos_column_test = filterColumn(test_sentences, pos_position)
    ner_column_test = filterColumn(test_sentences, ner_position)

    events_label2Idx, events_idx2Label = DatasetExtender.getDict(label_column_train)
    # there is a tag in the test file which does not appear in the train file
    # so the dictionaries have to be updated in order not to get an error
    test_label_dicts = GermEvalReader.getLabelDict(events_testFile, 2)
    for tag in test_label_dicts[0]:
        if tag not in events_label2Idx:
            events_label2Idx[tag] = len(events_label2Idx)
    events_idx2Label = {v: k for k, v in events_label2Idx.items()}

    events_pos2Idx, events_idx2pos = DatasetExtender.getDict(pos_column_train, withAddLabels=True)
    events_ner2Idx, events_ner2pos = DatasetExtender.getDict(ner_column_train, withAddLabels=True)

    # convert value to index
    words_train = GermEvalReader.convertValue2Idx(word_column_train, word2Idx, GermEvalReader.wordConverter)
    pos_train = GermEvalReader.convertValue2Idx(pos_column_train, events_pos2Idx, GermEvalReader.wordConverter)
    ner_train = GermEvalReader.convertValue2Idx(ner_column_train, events_ner2Idx, GermEvalReader.wordConverter)
    casing_train = GermEvalReader.convertValue2Idx(word_column_train, case2Idx, GermEvalReader.getCasing)
    labels_train = GermEvalReader.convertValue2Idx(label_column_train, events_label2Idx, GermEvalReader.labelConverter)

    words_dev = GermEvalReader.convertValue2Idx(word_column_dev, word2Idx, GermEvalReader.wordConverter)
    pos_dev = GermEvalReader.convertValue2Idx(pos_column_dev, events_pos2Idx, GermEvalReader.wordConverter)
    ner_dev = GermEvalReader.convertValue2Idx(ner_column_dev, events_ner2Idx, GermEvalReader.wordConverter)
    casing_dev = GermEvalReader.convertValue2Idx(word_column_dev, case2Idx, GermEvalReader.getCasing)
    labels_dev = GermEvalReader.convertValue2Idx(label_column_dev, events_label2Idx, GermEvalReader.labelConverter)

    words_test = GermEvalReader.convertValue2Idx(word_column_test, word2Idx, GermEvalReader.wordConverter)
    pos_test = GermEvalReader.convertValue2Idx(pos_column_test, events_pos2Idx, GermEvalReader.wordConverter)
    ner_test = GermEvalReader.convertValue2Idx(ner_column_test, events_ner2Idx, GermEvalReader.wordConverter)
    casing_test = GermEvalReader.convertValue2Idx(word_column_test, case2Idx, GermEvalReader.getCasing)
    labels_test = GermEvalReader.convertValue2Idx(label_column_test, events_label2Idx, GermEvalReader.labelConverter)

    # create numpy datasets
    events_train_x = GermEvalReader.createNumpyArray(words_train, windowSize, word2Idx)
    events_train_pos_x = GermEvalReader.createNumpyArray(pos_train, windowSize, events_pos2Idx)
    events_train_ner_x = GermEvalReader.createNumpyArray(ner_train, windowSize, events_ner2Idx)
    events_train_casing_x = GermEvalReader.createNumpyArray(casing_train, windowSize, case2Idx)
    events_train_y = np.concatenate(labels_train)

    events_dev_x = GermEvalReader.createNumpyArray(words_dev, windowSize, word2Idx)
    events_dev_pos_x = GermEvalReader.createNumpyArray(pos_dev, windowSize, events_pos2Idx)
    events_dev_ner_x = GermEvalReader.createNumpyArray(ner_dev, windowSize, events_ner2Idx)
    events_dev_casing_x = GermEvalReader.createNumpyArray(casing_dev, windowSize, case2Idx)
    events_dev_y = np.concatenate(labels_dev)

    events_test_x = GermEvalReader.createNumpyArray(words_test, windowSize, word2Idx)
    events_test_pos_x = GermEvalReader.createNumpyArray(pos_test, windowSize, events_pos2Idx)
    events_test_ner_x = GermEvalReader.createNumpyArray(ner_test, windowSize, events_ner2Idx)
    events_test_casing_x = GermEvalReader.createNumpyArray(casing_test, windowSize, case2Idx)
    events_test_y = np.concatenate(labels_test)

    print "shape of events_train_x:", events_train_x.shape
    print events_train_x[0]

    print "shape of events_train_pos_x:", events_train_pos_x.shape
    print events_train_pos_x[0]

    print "shape of events_train_ner_x:", events_train_ner_x.shape
    print events_train_ner_x[0]

    print "shape of events_train_casing_x:", events_train_casing_x.shape
    print events_train_casing_x[0]

    print "shape of events_train_y:", events_train_y.shape
    print events_train_y



    print "shape of events_dev_x:", events_dev_x.shape
    print events_dev_x[0]

    print "shape of events_dev_pos_x:", events_dev_pos_x.shape
    print events_dev_pos_x[0]

    print "shape of events_dev_ner_x:", events_dev_ner_x.shape
    print events_dev_ner_x[0]

    print "shape of events_dev_casing_x:", events_dev_casing_x.shape
    print events_dev_casing_x[0]

    print "shape of events_dev_y:", events_dev_y.shape
    print events_dev_y



    print "shape of events_test_x:", events_test_x.shape
    print events_test_x[0]

    print "shape of events_test_pos_x:", events_test_pos_x.shape
    print events_test_pos_x[0]

    print "shape of events_test_ner_x:", events_test_ner_x.shape
    print events_test_ner_x[0]

    print "shape of events_test_casing_x:", events_test_casing_x.shape
    print events_test_casing_x[0]

    print "shape of events_test_y:", events_test_y.shape
    print events_test_y



    input_train = [events_train_x, events_train_pos_x, events_train_ner_x, events_train_casing_x]
    input_dev = [events_dev_x, events_dev_pos_x, events_dev_ner_x, events_dev_casing_x]
    input_test = [events_test_x, events_test_pos_x, events_test_ner_x, events_test_casing_x]

    events_train_y_cat = np_utils.to_categorical(events_train_y, len(events_label2Idx))

    dicts = [word2Idx, events_pos2Idx, events_ner2Idx, case2Idx, events_label2Idx, events_idx2Label]
    return [input_train, events_train_y_cat], [input_dev, events_dev_y], [input_test, events_test_y], dicts

def filterColumn(sentences, position):
    return map(lambda sentence: sentence[:, position], sentences)

def extendDataset(filename, train_extensions, dev_extensions, test_extensions):
    train_sentences = GermEvalReader.readFile(events_trainFile, 0, 2)
    dev_sentences = GermEvalReader.readFile(events_devFile, 0, 2)
    test_sentences = GermEvalReader.readFile(events_testFile, 0, 2)

    filename, file_extension = path.splitext(filename)

    DatasetExtender.extendDataset("{0}_train_ext{1}".format(filename, file_extension), train_sentences, train_extensions)
    DatasetExtender.extendDataset("{0}_dev_ext{1}".format(filename, file_extension), dev_sentences, dev_extensions)
    DatasetExtender.extendDataset("{0}_test_ext{1}".format(filename, file_extension), test_sentences, test_extensions)

def getLabelDict():
    return GermEvalReader.getLabelDict(events_trainFile, 2)