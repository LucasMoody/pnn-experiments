import numpy as np

def measureAccuracy(predictions, dataset_y):
    return np.sum(predictions == dataset_y) / float(len(dataset_y))


def create_compute_BIOf1(idx2Label):
    return lambda predictions, dataset_y: compute_BIOf1(predictions, dataset_y, idx2Label)

def create_compute_IOf1(idx2Label):
    return lambda predictions, dataset_y: compute_IOf1(predictions, dataset_y, idx2Label)

# Method to compute the accruarcy. Call predict_labels to get the labels for the dataset
def compute_BIOf1(predictions, dataset_y, idx2Label):
    label_y = [idx2Label[element] for element in dataset_y]
    pred_labels = [idx2Label[element] for element in predictions]

    prec = compute_BIO_precision(pred_labels, label_y)
    rec = compute_BIO_precision(label_y, pred_labels)

    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);

    return prec, rec, f1

def compute_IOf1(predictions, dataset_y, idx2Label):
    label_y = np.asarray([idx2Label[element] for element in dataset_y])
    pred_labels = np.asarray([idx2Label[element] for element in predictions])
    label_y = label_y[0:30]
    pred_labels = pred_labels[0:30]
    prec = compute_IO_precision(pred_labels, label_y)
    rec = compute_IO_precision(label_y, pred_labels)

    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);

    return prec, rec, f1

def compute_IO_precision(guessed, correct):
    filterO = guessed != 'O'
    filtered_guessed = guessed[filterO]
    filtered_correct = correct[filterO]
    return np.sum(filtered_guessed == filtered_correct) / float(len(filtered_guessed))

def compute_BIO_precision(guessed, correct):
    correctCount = 0
    count = 0

    idx = 0
    while idx < len(guessed):
        if guessed[idx][0] == 'B':  # A new chunk starts
            count += 1

            if guessed[idx] == correct[idx]:
                idx += 1
                correctlyFound = True

                while idx < len(guessed) and guessed[idx][0] == 'I':  # Scan until it no longer starts with I
                    if guessed[idx] != correct[idx]:
                        correctlyFound = False

                    idx += 1

                if idx < len(guessed):
                    if correct[idx][0] == 'I':  # The chunk in correct was longer
                        correctlyFound = False

                if correctlyFound:
                    correctCount += 1
            else:
                idx += 1
        else:
            idx += 1

    precision = 0
    if count > 0:
        precision = float(correctCount) / count

    return precision

def calculateMatrix(prediction, observation, numberOfClasses):
    comparison = prediction == observation
    acc_ppv = 0
    acc_fdr = 0
    acc_for = 0
    acc_npv = 0
    acc_tpr = 0
    acc_fpr = 0
    acc_tp = 0
    acc_fp = 0
    acc_tn = 0
    acc_fn = 0

    for i in xrange(numberOfClasses):
        positives = comparison[prediction == i]
        negatives = comparison[prediction != i]
        true_positives = np.sum(positives)
        false_positives = np.sum(positives == False)
        true_negatives = np.sum(negatives)
        false_negatives = np.sum(negatives == False)
        if(len(positives) == 0):
            ppv = 1
            fdr = 0
        else:
            ppv = true_positives / float(len(positives))
            fdr = false_positives / float(len(positives))
        if(len(negatives) == 0):
            faor = 0
            npv = 1
        else:
            faor = true_negatives / float(len(negatives))
            npv = false_negatives / float(len(negatives))
        tpr = true_positives / float(true_positives + false_negatives)
        fpr = false_positives / float(false_positives + true_negatives)

        acc_ppv += ppv
        acc_fdr += fdr
        acc_for += faor
        acc_npv += npv
        acc_tpr += tpr
        acc_fpr += fpr

        acc_tp += true_positives
        acc_fp += false_positives
        acc_tn += true_negatives
        acc_fn += false_negatives

    acc_ppv /= float(numberOfClasses)
    acc_fdr /= float(numberOfClasses)
    acc_for /= float(numberOfClasses)
    acc_npv /= float(numberOfClasses)
    acc_tpr /= float(numberOfClasses)
    acc_fpr /= float(numberOfClasses)

    accuracy = (acc_tp + acc_tn) / float(numberOfClasses) / float(len(prediction))
    micro_precision = acc_tp / float(acc_tp + acc_fp)
    macro_precision = acc_ppv
    micro_recall = acc_tp / float(acc_tp + acc_fn)
    macro_recall = acc_tpr
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)

    print 'micro_precision', micro_precision
    print 'micro_recall', micro_recall
    print 'macro_precision', macro_precision
    print 'macro_recall', macro_recall
    print 'micro_f1', micro_f1
    print 'macro_f1', macro_f1
    print 'accuracy', accuracy

    return micro_precision, micro_recall, macro_precision, macro_recall, micro_f1, macro_f1, accuracy