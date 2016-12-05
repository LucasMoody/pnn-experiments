import numpy as np

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

