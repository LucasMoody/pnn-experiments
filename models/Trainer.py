import time
import numpy as np
import Sampler
import config

#sample_fun = Sampler.sampleEqualRanges
#sample_fun = Sampler.sampleLog2Ranges
#sample_fun = Sampler.sampleLog2AndEqualRanges
sample_fun = Sampler.samplePNNRanges

no_samples = config.number_of_samples
early_stopping_strike = 10

def trainModel(model, X_train, Y_train, number_of_epochs, minibatch_size, X_dev, Y_dev, X_test, Y_test, measurements=[], all_X_train = [], all_Y_train=[]):
    print "%d epochs" % number_of_epochs
    print "%d mini batches" % (len(X_train[0]) / minibatch_size)

    best_train_scores = [(0, 0) for i in xrange(len(measurements))]
    best_dev_scores = [(0, 0) for i in xrange(len(measurements))]
    best_test_scores = [(0, 0) for i in xrange(len(measurements))]
    best_model_weights = map(lambda x: x.copy(), model.get_weights())
    for epoch in xrange(number_of_epochs):
        start_time = time.time()
        #model.optimizer.lr.set_value(0.01)
        # default waere 0.001 bei adam
        # 1. und 2. epoche mit 0.01
        # 3. 4. 0.005
        # Rest 0.001
        if epoch == 0:
            model.optimizer.lr.set_value(0.01)
        if epoch == 2:
            model.optimizer.lr.set_value(0.005)
        if epoch == 4:
            model.optimizer.lr.set_value(0.001)
        model.fit(X_train, Y_train, nb_epoch=1, batch_size=minibatch_size, verbose=0, shuffle=False)

        print "%.2f sec for training" % (time.time() - start_time)
        if(len(all_X_train) == 0):
            pred_train = model.predict(X_train, verbose=0).argmax(axis=-1)  # Prediction of the classes
        else:
            pred_train = model.predict(all_X_train, verbose=0).argmax(axis=-1)  # Prediction of the classes
        pred_dev = model.predict(X_dev, verbose=0).argmax(axis=-1)  # Prediction of the classes
        pred_test = model.predict(X_test, verbose=0).argmax(axis=-1)  # test_case_x
        if(len(all_Y_train) == 0):
            measurements_train = map(lambda func: func(pred_train, Y_train.argmax(axis=1)), measurements)
        else:
            measurements_train = map(lambda func: func(pred_train, all_Y_train.argmax(axis=1)), measurements)
        measurements_dev = map(lambda func: func(pred_dev, Y_dev), measurements)
        measurements_test = map(lambda func: func(pred_test, Y_test), measurements)
        # update best scores
        for i in xrange(len(best_dev_scores)):
            score_train = measurements_train[i]
            score_dev = measurements_dev[i]
            score_test = measurements_test[i]
            if score_dev > best_dev_scores[i][0]:
                best_train_scores[i] = (score_train, epoch)
                best_dev_scores[i] = (score_dev, epoch)
                best_test_scores[i] = (score_test, epoch)
                best_model_weights = map(lambda x: x.copy(), model.get_weights())

        print 'Current dev_score:', measurements_dev[0]
        # early stopping
        best_dev_score_epoch = best_dev_scores[0][1]
        if epoch - best_dev_score_epoch > early_stopping_strike:
            break
    print 'Weight sum before settings best epoch:', reduce(lambda a, b: a + np.sum(b), model.get_weights(), 0)
    model.set_weights(best_model_weights)
    print 'Weight sum after finished training:', reduce(lambda a, b: a + np.sum(b), model.get_weights(), 0)
    print 'best dev score: {0} in epoch: {1}'.format(best_dev_scores[0][0], best_dev_scores[0][1])
    return best_train_scores, best_dev_scores, best_test_scores

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

def trainModelWithIncreasingData(model, X_train, Y_train, number_of_epochs, minibatch_size, X_dev, Y_dev, X_test, Y_test, measurements=[]):
    ranges = sample_fun(X_train, no_samples)

    train_scores = []
    dev_scores = []
    test_scores = []

    print "%d samples" % no_samples
    weights = map(lambda x: x.copy(), model.get_weights())

    for sample in ranges:
        print "Current sample: 0:%d" % sample
        start_time = time.time()

        sampled_train_x = map(lambda x: x[0:sample], X_train)
        sampled_train_y = Y_train[0:sample]
        # todo print sum of weights
        print 'Weight sum before training', reduce(lambda a, b: a + np.sum(b), model.get_weights(), 0)
        best_train_scores, best_dev_scores, best_test_scores = trainModel(model, sampled_train_x, sampled_train_y, number_of_epochs, minibatch_size, X_dev, Y_dev, X_test, Y_test, measurements=measurements, all_X_train=X_train, all_Y_train=Y_train)
        model.set_weights(weights)
        print 'Weight sum after resetting', reduce(lambda a, b: a + np.sum(b), model.get_weights(), 0)
        print "%.2f sec for sample training" % (time.time() - start_time)
        train_scores.append((best_train_scores, sample))
        dev_scores.append((best_dev_scores, sample))
        test_scores.append((best_test_scores, sample))

    return train_scores, dev_scores, test_scores