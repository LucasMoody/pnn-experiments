from datasets.ecbplus_ed import ECBPlusED
from datasets.tempeval3_ed import TempevalED
from datasets.ace_ed import ACEED
from datasets.tac2015_ed import TACED
from datasets.wsj_pos import WSJPos
from datasets.universal_dependencies_pos import UDPos
from datasets.conll_ner import CoNLLNer
from datasets.conll_chunking import CoNLLChunking

from embeddings.dependency_based_word_embeddings import DependencyBasedWordEmbeddings as Embeddings
import numpy as np

#Casing matrix
case2Idx = {
    'numeric': 0,
    'allLower': 1,
    'allUpper': 2,
    'initialUpper': 3,
    'other': 4,
    'mainly_numeric': 5,
    'contains_digit': 6,
    'PADDING': 7
}
n_in_case = len(case2Idx)

# Read in embeddings
embeddings = Embeddings.embeddings
word2Idx = Embeddings.word2Idx

def calcDescriptiveStats(reader):
    [input_train, train_y_cat], [input_dev,
                                 dev_y], [input_test, test_y], dicts = reader(
        3, word2Idx, case2Idx)
    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, caseLookup, label2Idx, idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    print train_x.shape[0], ' train samples'
    print train_x.shape[1], ' train dimension'
    print test_x.shape[0], ' test samples'

    train_y = train_y_cat.argmax(axis=-1)
    print 'train'
    computeFrequencies(train_y, idx2Label)
    print 'dev'
    computeFrequencies(dev_y, idx2Label)
    print 'test'
    computeFrequencies(test_y, idx2Label)


    return ''

def computeFrequencies(y, idx2Label):
    unique, counts = np.unique(y, return_counts=True)
    for idx, labelIdx in enumerate(unique):
        print "{}\t{}".format(idx2Label[labelIdx], counts[idx])


print 'ECB..............'
calcDescriptiveStats(ECBPlusED.readDataset)
print '\n--------------------\n'
print 'Tempeval..............'
calcDescriptiveStats(TempevalED.readDataset)
print '\n--------------------\n'
print 'Ace..............'
calcDescriptiveStats(ACEED.readDataset)
print '\n--------------------\n'
print 'Tac..............'
calcDescriptiveStats(TACED.readDataset)
print '\n--------------------\n'
print 'WSJ..............'
calcDescriptiveStats(WSJPos.readDataset)
print '\n--------------------\n'
print 'UD..............'
calcDescriptiveStats(UDPos.readDataset)
print '\n--------------------\n'
print 'NER..............'
calcDescriptiveStats(CoNLLNer.readDataset)
print '\n--------------------\n'
print 'Chunking..............'
calcDescriptiveStats(CoNLLChunking.readDataset)
