from .. import GermEvalReader
from .. import GermEvalReader_with_casing
from keras.utils import np_utils

ner_trainFile = 'datasets/conll_ner/data/eng.train'
ner_devFile = 'datasets/conll_ner/data/eng.testa'
ner_testFile = 'datasets/conll_ner/data/eng.testb'

#todo decide whether to put this as a parameter
maxTrainSentences = 20

def readDataset(windowSize, word2Idx, caseLookup):

    # Read in data
    print "Read in data and create matrices"
    ner_train_sentences = GermEvalReader.readFile(ner_trainFile, 0, 3, maxTrainSentences)
    ner_dev_sentences = GermEvalReader.readFile(ner_devFile, 0, 3)
    ner_test_sentences = GermEvalReader.readFile(ner_testFile, 0, 3)

    #Label mapping for POS
    ner_label2Idx, ner_idx2Label = GermEvalReader.getLabelDict(ner_trainFile)

    # Create numpy arrays
    ner_train_x, ner_train_case_x, ner_train_y = GermEvalReader_with_casing.createNumpyArrayWithCasing(
        ner_train_sentences, windowSize, word2Idx, ner_label2Idx, caseLookup)
    ner_dev_x, ner_dev_case_x, ner_dev_y = GermEvalReader_with_casing.createNumpyArrayWithCasing(ner_dev_sentences,
                                                                                                 windowSize, word2Idx,
                                                                                                 ner_label2Idx,
                                                                                                 caseLookup)
    ner_test_x, ner_test_case_x, ner_test_y = GermEvalReader_with_casing.createNumpyArrayWithCasing(ner_test_sentences,
                                                                                                    windowSize,
                                                                                                    word2Idx,
                                                                                                    ner_label2Idx,
                                                                                                    caseLookup)

    ner_train_y_cat = np_utils.to_categorical(ner_train_y, len(ner_label2Idx))

    return (ner_train_x, ner_train_case_x, ner_train_y, ner_train_y_cat), (ner_dev_x, ner_dev_case_x, ner_dev_y), (
    ner_test_x, ner_test_case_x, ner_test_y)