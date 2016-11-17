import numpy as np
from keras.callbacks import LambdaCallback

def createBatchCallback(model, X_dev, Y_dev, X_test, Y_test, dev_scores, test_scores):

    #def log_lambda_wrapper(model, X_dev, Y_dev, X_test, Y_test, dev_scores, test_scores):
    def log_lambda(batch, logs):
        print '\n', 'Current batch:', batch
        print 'log: ', logs
        printMetrics(model, X_dev, Y_dev, X_test, Y_test, dev_scores, test_scores)
    return LambdaCallback(on_batch_end=log_lambda)
    #return log_lambda_wrapper

def printMetrics(model, X_dev, Y_dev, X_test, Y_test, dev_scores, test_scores):
    pred_dev = model.predict(X_dev, verbose=0).argmax(axis=-1)  # Prediction of the classes
    dev_acc = np.sum(pred_dev == Y_dev) / float(len(Y_dev))
    dev_scores.append(dev_acc)
    pred_test = model.predict(X_test, verbose=0).argmax(axis=-1)  # test_case_x
    test_acc = np.sum(pred_test == Y_test) / float(len(Y_test))
    test_scores.append(test_acc)

    print "Accuracy dev: %.2f%%" % ((dev_acc * 100))
    print "Accuracy test: %.2f%%" % ((test_acc * 100))

def createBatchTrainAccCallback(train_scores):
    def log_lambda(batch, logs):
        #print '\n', 'Current batch:', batch
        #print 'log: ', logs
        train_scores.append(logs['acc'])
    return LambdaCallback(on_batch_end=log_lambda)