import time
import numpy as np
import Sampler
import config
from plots import LearningCurve
from logs import Logger
#sample_fun = Sampler.sampleEqualRanges
#sample_fun = Sampler.sampleLog2Ranges
#sample_fun = Sampler.sampleLog2AndEqualRanges
sample_fun = Sampler.sampleSimplePNNRanges

no_samples = config.number_of_samples
early_stopping_strike = 10

def trainModel(model, X_train, Y_train, number_of_epochs, minibatch_size, X_dev, Y_dev, X_test, Y_test, measurements=[]):
    print "%d epochs" % number_of_epochs
    print "%d mini batches" % (len(X_train[0]) / minibatch_size)

    best_train_scores = [(-1, 0) for i in xrange(len(measurements))]
    best_dev_scores = [(-1, 0) for i in xrange(len(measurements))]
    best_test_scores = [(-1, 0) for i in xrange(len(measurements))]
    best_model_weights = map(lambda x: x.copy(), model.get_weights())

    dev_scores = []
    train_scores = []
    # ----- Training ---- #
    for epoch in xrange(number_of_epochs):
        start_time = time.time()
        if epoch == 0:
            model.optimizer.lr.set_value(0.01)
        if epoch == 2:
            model.optimizer.lr.set_value(0.005)
        if epoch == 4:
            model.optimizer.lr.set_value(0.001)
        model.fit(X_train, Y_train, nb_epoch=1, batch_size=minibatch_size, verbose=0, shuffle=True)

        print "%.2f sec for training" % (time.time() - start_time)
        # only dev scores need to be calculated
        pred_dev = model.predict(X_dev, verbose=0).argmax(axis=-1)  # Prediction of the classes
        measurements_dev = map(lambda func: func(pred_dev, Y_dev), measurements)
        pred_train = model.predict(X_train, verbose=0).argmax(axis=-1)  # Prediction of the classes
        measurements_train = map(lambda func: func(pred_train, Y_train.argmax(axis=1)), measurements)
        # update best scores
        for i in xrange(len(best_dev_scores)):
            # compare dev scores to get best one
            score_dev = measurements_dev[i]
            if score_dev > best_dev_scores[i][0]:
                best_dev_scores[i] = (score_dev, epoch)
                best_model_weights = map(lambda x: x.copy(), model.get_weights())

        #print 'Current dev_score: {0:.4f} and current patience: {1}'.format(measurements_dev[0] * 100, epoch - best_dev_scores[0][1])
        print 'Current train_score/dev_score: {0:.4f}/{1:.4f} and current patience: {2}'.format(measurements_train[0] * 100, measurements_dev[0] * 100, epoch - best_dev_scores[0][1])
        dev_scores.append(measurements_dev[0] * 100)
        train_scores.append(measurements_train[0] * 100)
        # early stopping
        best_dev_score_epoch = best_dev_scores[0][1]
        if epoch - best_dev_score_epoch > early_stopping_strike:
            break



    Logger.save_scores_for_overfitting(train_scores, dev_scores)

    # ----- score calculations and weight resetting to best score ----- #
    # set back weights to best epoch
    print 'Weight sum before setting best epoch:', reduce(lambda a, b: a + np.sum(b), model.get_weights(), 0)
    model.set_weights(best_model_weights)
    print 'Weight sum after finished training:', reduce(lambda a, b: a + np.sum(b), model.get_weights(), 0)

    # calculate train and test scores for best epoch as well
    for i in xrange(len(best_dev_scores)):
        # make predictions on other datasets
        # if not train set is not the full train dataset
        pred_train = model.predict(X_train, verbose=0).argmax(axis=-1)  # Prediction of the classes
        pred_dev = model.predict(X_dev, verbose=0).argmax(axis=-1)  # Prediction of the classes
        pred_test = model.predict(X_test, verbose=0).argmax(axis=-1)  # test_case_x
        # calculate scores of predictions
        measurements_train = map(lambda func: func(pred_train, Y_train.argmax(axis=1)), measurements)
        measurements_dev = map(lambda func: func(pred_dev, Y_dev), measurements)
        measurements_test = map(lambda func: func(pred_test, Y_test), measurements)

        # calculate other scores
        score_train = measurements_train[i]
        score_dev = measurements_dev[i]
        score_test = measurements_test[i]

        # test whether earlier calculated dev score is the same as with reset weights
        if best_dev_scores[i][0] != score_dev:
            raise ValueError('Newly calculated best score should be the same as earlier saved one!')
        # assign new best scores
        best_dev_epoch = best_dev_scores[i][1]
        best_train_scores[i] = (score_train, best_dev_epoch)
        best_test_scores[i] = (score_test, best_dev_epoch)
    print 'best dev score: {0} in epoch: {1}'.format(best_dev_scores[0][0], best_dev_scores[0][1])
    return best_train_scores, best_dev_scores, best_test_scores

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
        best_train_scores, best_dev_scores, best_test_scores = trainModel(model, sampled_train_x, sampled_train_y, number_of_epochs, minibatch_size, X_dev, Y_dev, X_test, Y_test, measurements=measurements)
        model.set_weights(weights)
        print 'Weight sum after resetting', reduce(lambda a, b: a + np.sum(b), model.get_weights(), 0)
        print "%.2f sec for sample training" % (time.time() - start_time)
        train_scores.append((best_train_scores, sample))
        dev_scores.append((best_dev_scores, sample))
        test_scores.append((best_test_scores, sample))

    return train_scores, dev_scores, test_scores