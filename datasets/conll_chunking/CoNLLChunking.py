from datasets import GermEvalReader, DatasetExtender
from keras.utils import np_utils
from os import path
import numpy as np

chunk_trainFile = 'datasets/conll_chunking/data/train.txt'
chunk_testFile = 'datasets/conll_chunking/data/test.txt'

trainFileExt = 'datasets/conll_chunking/data/chunking_train_ext.conllu'
devFileExt = 'datasets/conll_chunking/data/chunking_dev_ext.conllu'
testFileExt = 'datasets/conll_chunking/data/chunking_test_ext.conllu'

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
    chunk_train_sentences = GermEvalReader.readFile(chunk_trainFile, 0, 2)
    chunk_all_test_sentences = GermEvalReader.readFile(chunk_testFile, 0, 2)
    chunk_dev_sentences = chunk_all_test_sentences[len(chunk_all_test_sentences) / 2:]
    chunk_test_sentences = chunk_all_test_sentences[:len(chunk_all_test_sentences)/2]

    #Label mapping for POS
    chunk_label2Idx, chunk_idx2Label = GermEvalReader.getLabelDict(chunk_trainFile, 2)
    # there is a tag in the test file which does not appear in the train file
    # so the dictionaries have to be updated in order not to get an error
    test_label_dicts = GermEvalReader.getLabelDict(chunk_testFile, 2)
    for tag in test_label_dicts[0]:
        if tag not in chunk_label2Idx:
            chunk_label2Idx[tag] = len(chunk_label2Idx)
    chunk_idx2Label = {v: k for k, v in chunk_label2Idx.items()}

    # Create numpy arrays
    chunk_train_x, chunk_train_case_x, chunk_train_y = GermEvalReader.createNumpyArrayWithCasing(
        chunk_train_sentences, windowSize, word2Idx, chunk_label2Idx, case2Idx)
    chunk_dev_x, chunk_dev_case_x, chunk_dev_y = GermEvalReader.createNumpyArrayWithCasing(chunk_dev_sentences,
                                                                                     windowSize, word2Idx,
                                                                                     chunk_label2Idx,
                                                                                     case2Idx)
    chunk_test_x, chunk_test_case_x, chunk_test_y = GermEvalReader.createNumpyArrayWithCasing(chunk_test_sentences,
                                                                                        windowSize,
                                                                                        word2Idx,
                                                                                        chunk_label2Idx,
                                                                                        case2Idx)
    chunk_input_train = [chunk_train_x, chunk_train_case_x]
    chunk_input_dev = [chunk_dev_x, chunk_dev_case_x]
    chunk_input_test = [chunk_test_x, chunk_test_case_x]

    chunk_train_y_cat = np_utils.to_categorical(chunk_train_y, len(chunk_label2Idx))

    chunk_dicts = [word2Idx, case2Idx, chunk_label2Idx, chunk_idx2Label]
    return [chunk_input_train, chunk_train_y_cat], [chunk_input_dev, chunk_dev_y], [chunk_input_test, chunk_test_y], chunk_dicts

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

    chunking_label2Idx, chunking_idx2Label = DatasetExtender.getDict(label_column_train)
    # there is a tag in the test file which does not appear in the train file
    # so the dictionaries have to be updated in order not to get an error
    test_label_dicts = GermEvalReader.getLabelDict(chunk_testFile, 2)
    for tag in test_label_dicts[0]:
        if tag not in chunking_label2Idx:
            chunking_label2Idx[tag] = len(chunking_label2Idx)
    chunking_idx2Label = {v: k for k, v in chunking_label2Idx.items()}

    chunking_pos2Idx, chunking_idx2pos = DatasetExtender.getDict(pos_column_train, withAddLabels=True)
    chunking_ner2Idx, chunking_ner2pos = DatasetExtender.getDict(ner_column_train, withAddLabels=True)

    # convert value to index
    words_train = GermEvalReader.convertValue2Idx(word_column_train, word2Idx, GermEvalReader.wordConverter)
    pos_train = GermEvalReader.convertValue2Idx(pos_column_train, chunking_pos2Idx, GermEvalReader.wordConverter)
    ner_train = GermEvalReader.convertValue2Idx(ner_column_train, chunking_ner2Idx, GermEvalReader.wordConverter)
    casing_train = GermEvalReader.convertValue2Idx(word_column_train, case2Idx, GermEvalReader.getCasing)
    labels_train = GermEvalReader.convertValue2Idx(label_column_train, chunking_label2Idx, GermEvalReader.labelConverter)

    words_dev = GermEvalReader.convertValue2Idx(word_column_dev, word2Idx, GermEvalReader.wordConverter)
    pos_dev = GermEvalReader.convertValue2Idx(pos_column_dev, chunking_pos2Idx, GermEvalReader.wordConverter)
    ner_dev = GermEvalReader.convertValue2Idx(ner_column_dev, chunking_ner2Idx, GermEvalReader.wordConverter)
    casing_dev = GermEvalReader.convertValue2Idx(word_column_dev, case2Idx, GermEvalReader.getCasing)
    labels_dev = GermEvalReader.convertValue2Idx(label_column_dev, chunking_label2Idx, GermEvalReader.labelConverter)

    words_test = GermEvalReader.convertValue2Idx(word_column_test, word2Idx, GermEvalReader.wordConverter)
    pos_test = GermEvalReader.convertValue2Idx(pos_column_test, chunking_pos2Idx, GermEvalReader.wordConverter)
    ner_test = GermEvalReader.convertValue2Idx(ner_column_test, chunking_ner2Idx, GermEvalReader.wordConverter)
    casing_test = GermEvalReader.convertValue2Idx(word_column_test, case2Idx, GermEvalReader.getCasing)
    labels_test = GermEvalReader.convertValue2Idx(label_column_test, chunking_label2Idx, GermEvalReader.labelConverter)

    # create numpy datasets
    chunking_train_x = GermEvalReader.createNumpyArray(words_train, windowSize, word2Idx)
    chunking_train_pos_x = GermEvalReader.createNumpyArray(pos_train, windowSize, chunking_pos2Idx)
    chunking_train_ner_x = GermEvalReader.createNumpyArray(ner_train, windowSize, chunking_ner2Idx)
    chunking_train_casing_x = GermEvalReader.createNumpyArray(casing_train, windowSize, case2Idx)
    chunking_train_y = np.concatenate(labels_train)

    chunking_dev_x = GermEvalReader.createNumpyArray(words_dev, windowSize, word2Idx)
    chunking_dev_pos_x = GermEvalReader.createNumpyArray(pos_dev, windowSize, chunking_pos2Idx)
    chunking_dev_ner_x = GermEvalReader.createNumpyArray(ner_dev, windowSize, chunking_ner2Idx)
    chunking_dev_casing_x = GermEvalReader.createNumpyArray(casing_dev, windowSize, case2Idx)
    chunking_dev_y = np.concatenate(labels_dev)

    chunking_test_x = GermEvalReader.createNumpyArray(words_test, windowSize, word2Idx)
    chunking_test_pos_x = GermEvalReader.createNumpyArray(pos_test, windowSize, chunking_pos2Idx)
    chunking_test_ner_x = GermEvalReader.createNumpyArray(ner_test, windowSize, chunking_ner2Idx)
    chunking_test_casing_x = GermEvalReader.createNumpyArray(casing_test, windowSize, case2Idx)
    chunking_test_y = np.concatenate(labels_test)

    print "shape of chunking_train_x:", chunking_train_x.shape
    print chunking_train_x[0]

    print "shape of chunking_train_pos_x:", chunking_train_pos_x.shape
    print chunking_train_pos_x[0]

    print "shape of chunking_train_ner_x:", chunking_train_ner_x.shape
    print chunking_train_ner_x[0]

    print "shape of chunking_train_casing_x:", chunking_train_casing_x.shape
    print chunking_train_casing_x[0]

    print "shape of chunking_train_y:", chunking_train_y.shape
    print chunking_train_y



    print "shape of chunking_dev_x:", chunking_dev_x.shape
    print chunking_dev_x[0]

    print "shape of chunking_dev_pos_x:", chunking_dev_pos_x.shape
    print chunking_dev_pos_x[0]

    print "shape of chunking_dev_ner_x:", chunking_dev_ner_x.shape
    print chunking_dev_ner_x[0]

    print "shape of chunking_dev_casing_x:", chunking_dev_casing_x.shape
    print chunking_dev_casing_x[0]

    print "shape of chunking_dev_y:", chunking_dev_y.shape
    print chunking_dev_y



    print "shape of chunking_test_x:", chunking_test_x.shape
    print chunking_test_x[0]

    print "shape of chunking_test_pos_x:", chunking_test_pos_x.shape
    print chunking_test_pos_x[0]

    print "shape of chunking_test_ner_x:", chunking_test_ner_x.shape
    print chunking_test_ner_x[0]

    print "shape of chunking_test_casing_x:", chunking_test_casing_x.shape
    print chunking_test_casing_x[0]

    print "shape of chunking_test_y:", chunking_test_y.shape
    print chunking_test_y



    input_train = [chunking_train_x, chunking_train_pos_x, chunking_train_ner_x, chunking_train_casing_x]
    input_dev = [chunking_dev_x, chunking_dev_pos_x, chunking_dev_ner_x, chunking_dev_casing_x]
    input_test = [chunking_test_x, chunking_test_pos_x, chunking_test_ner_x, chunking_test_casing_x]

    chunking_train_y_cat = np_utils.to_categorical(chunking_train_y, len(chunking_label2Idx))

    dicts = [word2Idx, chunking_pos2Idx, chunking_ner2Idx, case2Idx, chunking_label2Idx, chunking_idx2Label]
    return [input_train, chunking_train_y_cat], [input_dev, chunking_dev_y], [input_test, chunking_test_y], dicts

def filterColumn(sentences, position):
    return map(lambda sentence: sentence[:, position], sentences)

def extendDataset(filename, train_extensions, dev_extensions, test_extensions):
    train_sentences = GermEvalReader.readFile(chunk_trainFile, 0, 2)
    all_test_sentences = GermEvalReader.readFile(chunk_testFile, 0, 2)
    dev_sentences = all_test_sentences[len(all_test_sentences) / 2:]
    test_sentences = all_test_sentences[:len(all_test_sentences) / 2]

    filename, file_extension = path.splitext(filename)

    DatasetExtender.extendDataset("{0}_train_ext{1}".format(filename, file_extension), train_sentences, train_extensions)
    DatasetExtender.extendDataset("{0}_dev_ext{1}".format(filename, file_extension), dev_sentences, dev_extensions)
    DatasetExtender.extendDataset("{0}_test_ext{1}".format(filename, file_extension), test_sentences, test_extensions)

def getLabelDict():
    return GermEvalReader.getLabelDict(chunk_trainFile)