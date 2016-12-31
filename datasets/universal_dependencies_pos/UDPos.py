from datasets import GermEvalReader
from keras.utils import np_utils

pos_trainFile = 'datasets/universal_dependencies_pos/data/en-ud-train.conllu'
pos_devFile = 'datasets/universal_dependencies_pos/data/en-ud-dev.conllu'
pos_testFile = 'datasets/universal_dependencies_pos/data/en-ud-test.conllu'

def readDataset(windowSize, word2Idx, caseLookup):

    # Read in data
    print "Read in data and create matrices"
    pos_train_sentences = GermEvalReader.readFile(pos_trainFile, 1, 3)
    pos_dev_sentences = GermEvalReader.readFile(pos_devFile, 1, 3)
    pos_test_sentences = GermEvalReader.readFile(pos_testFile, 1, 3)

    #Label mapping for POS
    pos_label2Idx, pos_idx2Label = GermEvalReader.getLabelDict(pos_trainFile)

    # Create numpy arrays
    pos_train_x, pos_train_case_x, pos_train_y = GermEvalReader.createNumpyArrayWithCasing(pos_train_sentences, windowSize, word2Idx, pos_label2Idx, caseLookup)
    pos_dev_x, pos_dev_case_x, pos_dev_y = GermEvalReader.createNumpyArrayWithCasing(pos_dev_sentences, windowSize, word2Idx, pos_label2Idx, caseLookup)
    pos_test_x, pos_test_case_x, pos_test_y = GermEvalReader.createNumpyArrayWithCasing(pos_test_sentences, windowSize, word2Idx, pos_label2Idx, caseLookup)

    '''pos_train_y_cat = np_utils.to_categorical(pos_train_y, len(pos_label2Idx))

    return (pos_train_x, pos_train_case_x, pos_train_y, pos_train_y_cat), (pos_dev_x, pos_dev_case_x, pos_dev_y), (pos_test_x, pos_test_case_x, pos_test_y)'''


    pos_input_train = [pos_train_x, pos_train_case_x]
    pos_input_dev = [pos_dev_x, pos_dev_case_x]
    pos_input_test = [pos_test_x, pos_test_case_x]

    pos_train_y_cat = np_utils.to_categorical(pos_train_y, len(pos_label2Idx))

    return [pos_input_train, pos_train_y_cat], [pos_input_dev, pos_dev_y], [pos_input_test, pos_test_y]

def getLabelDict():
    return GermEvalReader.getLabelDict(pos_trainFile)