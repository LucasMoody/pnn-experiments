import time
import numpy as np
import Sampler

sample_fun = Sampler.sampleEqualRanges
#sample_fun = Sampler.sampleLog2Ranges

no_samples = 5

def trainModel(model, X_train, Y_train, number_of_epochs, minibatch_size, X_dev, Y_dev, X_test, Y_test, callbacks=[]):
    print "%d epochs" % number_of_epochs
    print "%d mini batches" % (len(X_train[0]) / minibatch_size)

    best_dev_acc = 0
    best_test_acc = 0
    best_acc_epoch = 0
    best_dev_f1 = 0
    best_test_f1 = 0
    best_f1_epoch = 0
    for epoch in xrange(number_of_epochs):
        start_time = time.time()

        model.fit(X_train, Y_train, nb_epoch=1, batch_size=minibatch_size, verbose=0, shuffle=True)

        print "%.2f sec for training" % (time.time() - start_time)

        pred_dev = model.predict(X_dev, verbose=0).argmax(axis=-1)  # Prediction of the classes
        dev_acc = np.sum(pred_dev == Y_dev) / float(len(Y_dev))
        dev_micro_precision, dev_micro_recall, dev_macro_precision, dev_macro_recall, dev_micro_f1, dev_macro_f1, dev_accuracy = calculateMatrix(
            pred_dev, Y_dev, Y_train.shape[1])
        pred_test = model.predict(X_test, verbose=0).argmax(axis=-1)  # test_case_x
        test_micro_precision, test_micro_recall, test_macro_precision, test_macro_recall, test_micro_f1, test_macro_f1, test_accuracy = calculateMatrix(
            pred_test, Y_test, Y_train.shape[1])
        test_acc = np.sum(pred_test == Y_test) / float(len(Y_test))
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_test_acc = test_acc
            best_acc_epoch = epoch
        if dev_macro_f1 > best_dev_f1:
            best_dev_f1 = dev_macro_f1
            best_test_f1 = test_macro_f1
            best_f1_epoch = epoch

    print "Best accuracy epoch: %d" % best_acc_epoch
    print "Best accuracy dev: %.2f%%" % (best_dev_acc * 100)
    print "Best accuracy test: %.2f%%" % (best_test_acc * 100)
    print "Best f1 epoch: %d" % best_f1_epoch
    print "Best f1 dev: %.2f%%" % (best_dev_f1 * 100)
    print "Best f1 test: %.2f%%" % (best_test_f1 * 100)
    return best_dev_acc, best_test_acc, best_dev_f1, best_test_f1

def calculateAcc(prediction, observation):
    return np.sum(prediction == observation) / float(len(observation))

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

def trainModelWithIncreasingData(model, X_train, Y_train, number_of_epochs, minibatch_size, X_dev, Y_dev, X_test, Y_test, callbacks=[]):
    ranges = sample_fun(X_train, no_samples)

    dev_accs = []
    test_accs = []
    dev_f1s = []
    test_f1s = []

    print "%d samples" % no_samples

    for sample in ranges:
        print "Current sample: 0:%d" % sample
        start_time = time.time()

        sampled_train_x = map(lambda x: x[0:sample], X_train)
        sampled_train_y = Y_train[0:sample]

        best_dev_acc, best_test_acc, best_dev_f1, best_test_f1 = trainModel(model, sampled_train_x, sampled_train_y, number_of_epochs, minibatch_size, X_dev, Y_dev, X_test, Y_test)

        print "%.2f sec for sample training" % (time.time() - start_time)
        dev_accs.append(best_dev_acc)
        test_accs.append(best_test_acc)
        dev_f1s.append(best_dev_f1)
        test_f1s.append(best_test_f1)

    return dev_accs, test_accs, dev_f1s, test_f1s, ranges