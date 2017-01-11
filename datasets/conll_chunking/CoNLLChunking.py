from datasets import GermEvalReader, DatasetExtender
from keras.utils import np_utils
from os import path
import numpy as np

chunk_trainFile = 'datasets/conll_chunking/data/train.txt'
chunk_testFile = 'datasets/conll_chunking/data/test.txt'

ner_trainFileExt = 'datasets/conll_ner/data/eng_train_ext.conllu'
ner_devFileExt = 'datasets/conll_ner/data/eng_dev_ext.conllu'
ner_testFileExt = 'datasets/conll_ner/data/eng_test_ext.conllu'

word_position = 0
label_position = 1
pos_position = 2
other_positions = [2]
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
    casing_train = GermEvalReader.convertValue2Idx(word_column_train, case2Idx, GermEvalReader.getCasing)
    labels_train = GermEvalReader.convertValue2Idx(label_column_train, ner_label2Idx, GermEvalReader.labelConverter)

    words_dev = GermEvalReader.convertValue2Idx(word_column_dev, word2Idx, GermEvalReader.wordConverter)
    pos_dev = GermEvalReader.convertValue2Idx(pos_column_dev, ner_pos2Idx, GermEvalReader.wordConverter)
    casing_dev = GermEvalReader.convertValue2Idx(word_column_dev, case2Idx, GermEvalReader.getCasing)
    labels_dev = GermEvalReader.convertValue2Idx(label_column_dev, ner_label2Idx, GermEvalReader.labelConverter)

    words_test = GermEvalReader.convertValue2Idx(word_column_test, word2Idx, GermEvalReader.wordConverter)
    pos_test = GermEvalReader.convertValue2Idx(pos_column_test, ner_pos2Idx, GermEvalReader.wordConverter)
    casing_test = GermEvalReader.convertValue2Idx(word_column_test, case2Idx, GermEvalReader.getCasing)
    labels_test = GermEvalReader.convertValue2Idx(label_column_test, ner_label2Idx, GermEvalReader.labelConverter)

    # create numpy datasets
    ner_train_x = GermEvalReader.createNumpyArray(words_train, windowSize, word2Idx)
    ner_train_pos_x = GermEvalReader.createNumpyArray(pos_train, windowSize, ner_pos2Idx)
    ner_train_casing_x = GermEvalReader.createNumpyArray(casing_train, windowSize, case2Idx)
    ner_train_y = np.concatenate(labels_train)

    ner_dev_x = GermEvalReader.createNumpyArray(words_dev, windowSize, word2Idx)
    ner_dev_pos_x = GermEvalReader.createNumpyArray(pos_dev, windowSize, ner_pos2Idx)
    ner_dev_casing_x = GermEvalReader.createNumpyArray(casing_dev, windowSize, case2Idx)
    ner_dev_y = np.concatenate(labels_dev)

    ner_test_x = GermEvalReader.createNumpyArray(words_test, windowSize, word2Idx)
    ner_test_pos_x = GermEvalReader.createNumpyArray(pos_test, windowSize, ner_pos2Idx)
    ner_test_casing_x = GermEvalReader.createNumpyArray(casing_test, windowSize, case2Idx)
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

    dicts = [word2Idx, ner_pos2Idx, case2Idx, ner_label2Idx, ner_idx2Label]
    return [input_train, ner_train_y_cat], [input_dev, ner_dev_y], [input_test, ner_test_y], dicts

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