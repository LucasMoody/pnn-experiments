import time
import numpy as np

def trainModel(model, X_train, Y_train, number_of_epochs, minibatch_size, X_dev, Y_dev, X_test, Y_test, callbacks=[]):
    print "%d epochs" % number_of_epochs
    print "%d mini batches" % (len(X_train[0]) / minibatch_size)

    best_dev_score = 0
    best_test_score = 0
    #model_callbacks = map(lambda x: x(model, X_dev, Y_dev, X_test, Y_test, dev_scores, test_scores), callbacks)
    for epoch in xrange(number_of_epochs):
        start_time = time.time()

        model.fit(X_train, Y_train, nb_epoch=1, batch_size=minibatch_size, verbose=1,
                  shuffle=True, callbacks=callbacks)

        print "%.2f sec for training" % (time.time() - start_time)

        #pred_dev = model.predict(X_dev, verbose=0).argmax(axis=-1)  # Prediction of the classes
        #dev_acc = np.sum(pred_dev == Y_dev) / float(len(Y_dev))
        #dev_scores.append(dev_acc)
        #pred_test = model.predict(X_test, verbose=0).argmax(axis=-1)  # test_case_x
        #test_acc = np.sum(pred_test == Y_test) / float(len(Y_test))
        #test_scores.append(test_acc)
        #if dev_acc > best_dev_score:
        #    best_dev_score = dev_acc
        #    best_test_score = test_acc

        #print "Accuracy dev: %.2f%% (best: %.2f%%)" % ((dev_acc * 100), best_dev_score * 100)
        #print "Accuracy test: %.2f%% (best: %.2f%%)" % ((test_acc * 100), best_test_score * 100)

    #return dev_scores, test_scores