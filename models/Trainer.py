import time
import numpy as np
import Sampler

sample_fun = Sampler.sampleEqualRanges
no_samples = 2

def trainModel(model, X_train, Y_train, number_of_epochs, minibatch_size, X_dev, Y_dev, X_test, Y_test, callbacks=[]):
    print "%d epochs" % number_of_epochs
    print "%d mini batches" % (len(X_train[0]) / minibatch_size)

    best_dev_score = 0
    best_test_score = 0
    best_score_epoch = 0
    for epoch in xrange(number_of_epochs):
        start_time = time.time()

        model.fit(X_train, Y_train, nb_epoch=1, batch_size=minibatch_size, verbose=1,
                  shuffle=True, callbacks=callbacks)

        print "%.2f sec for training" % (time.time() - start_time)

        pred_dev = model.predict(X_dev, verbose=0).argmax(axis=-1)  # Prediction of the classes
        dev_acc = np.sum(pred_dev == Y_dev) / float(len(Y_dev))
        pred_test = model.predict(X_test, verbose=0).argmax(axis=-1)  # test_case_x
        test_acc = np.sum(pred_test == Y_test) / float(len(Y_test))
        if dev_acc > best_dev_score:
            best_dev_score = dev_acc
            best_test_score = test_acc
            best_score_epoch = epoch

    print "Best epoch: %d" % best_score_epoch
    print "Best accuracy dev: %.2f%%" % (best_dev_score * 100)
    print "Best accuracy test: %.2f%%" % (test_acc * 100)
    return best_dev_score, best_test_score


def trainModelWithIncreasingData(model, X_train, Y_train, number_of_epochs, minibatch_size, X_dev, Y_dev, X_test, Y_test, callbacks=[]):

    ranges = sample_fun(X_train, no_samples)
    dev_scores = [0]
    test_scores = [0]

    print "%d samples" % no_samples

    dev_scores = []
    test_scores = []
    for sample in ranges:
        print "Current sample: 0:%d" % sample
        start_time = time.time()

        sampled_train_x = map(lambda x: x[0:sample], X_train)
        sampled_train_y = Y_train[0:sample]

        best_dev_score, best_test_score = trainModel(model, sampled_train_x, sampled_train_y, number_of_epochs, minibatch_size, X_dev, Y_dev, X_test, Y_test,
                   callbacks=callbacks)

        print "%.2f sec for sample training" % (time.time() - start_time)
        dev_scores.append(best_dev_score)
        test_scores.append(best_test_score)

    return dev_scores, test_scores