import numpy as np
from datasets import GermEvalReader

def extendDataset(filename, sentences, extensions):
    overall_word_idx = 0
    with open(filename, 'w') as f:
        for sentence_idx in xrange(len(sentences)):
            cur_sentence = sentences[sentence_idx]
            for word_label_idx in xrange(len(cur_sentence)):
                [word, label] = cur_sentence[word_label_idx]
                f.write("{0}\t{1}".format(word, label))
                for extension in extensions:
                    f.write("\t{0}".format(extension[overall_word_idx]))
                f.write("\n")
                overall_word_idx += 1
            f.write("\n")
    print "{} number of words".format(overall_word_idx)

def getDict(column, withAddLabels=False):
    # join sentences
    column_con = np.concatenate(column)
    label2Idx = {}
    for value in column_con:
        if value not in label2Idx:
            label2Idx[value] = len(label2Idx)
    if withAddLabels:
        label2Idx['UNKNOWN'] = len(label2Idx)
        label2Idx['PADDING'] = len(label2Idx)
    idx2Label = {v: k for k, v in label2Idx.items()}
    return label2Idx, idx2Label

def createNumpyArraysForFeature(train_sentences, dev_sentences, test_sentences, position, converter, windowSize):
    column_train = filterColumn(train_sentences, position)
    column_dev = filterColumn(dev_sentences, position)
    column_test = filterColumn(test_sentences, position)

    label2Idx, Idx2label = getDict(column_train, withAddLabels=True)

    train = GermEvalReader.convertValue2Idx(column_train, label2Idx, converter)
    dev = GermEvalReader.convertValue2Idx(column_dev, label2Idx, converter)
    test = GermEvalReader.convertValue2Idx(column_test, label2Idx, converter)

    train_x = GermEvalReader.createNumpyArray(train, windowSize, label2Idx)
    dev_x = GermEvalReader.createNumpyArray(dev, windowSize, label2Idx)
    test_x = GermEvalReader.createNumpyArray(test, windowSize, label2Idx)

    return train_x, dev_x, test_x, label2Idx

def filterColumn(sentences, position):
    return map(lambda sentence: sentence[:, position], sentences)