from keras.layers import Input, Embedding, Flatten, Dense, merge
from keras.models import Model
import theano
import numpy as np
import time

#####################################
#
# Create the Keras Network for POS
#
#####################################
def buildPosModel(n_in, embeddings, n_in_case, numHiddenUnitsPOS, pos_n_out, metrics=[]):
    words_input = Input(shape=(n_in,), dtype='int32', name='words_input')
    wordEmbeddingLayer = Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in,  weights=[embeddings])
    words = wordEmbeddingLayer(words_input)
    words = Flatten(name='words_flatten')(words)

    caseMatrix = np.identity(n_in_case, dtype=theano.config.floatX)

    case_input = Input(shape=(n_in,), dtype='int32', name='case_input')
    caseEmbeddingLayer = Embedding(output_dim=caseMatrix.shape[1], input_dim=caseMatrix.shape[0], input_length=n_in, weights=[caseMatrix])
    casing = caseEmbeddingLayer(case_input)
    casing = Flatten(name='casing_flatten')(casing)

    words_casing_merged = merge([words, casing], mode='concat')
    pos_hidden_layer = Dense(numHiddenUnitsPOS, activation='tanh', name='pos_hidden')
    pos_hidden = pos_hidden_layer(words_casing_merged)

    pos_output_layer = Dense(output_dim=pos_n_out, activation='softmax', name='pos_output')
    pos_output = pos_output_layer(pos_hidden)

    model = Model(input=[words_input, case_input], output=[pos_output])

    #Don't update embeddings
    wordEmbeddingLayer.trainable_weights = []
    caseEmbeddingLayer.trainable_weights = []

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)

    print model.summary()

    return model

'''
def trainPosModel(model, X_train, Y_train, number_of_epochs, minibatch_size, X_dev, Y_dev, X_test, Y_test):
    print "%d epochs" % number_of_epochs
    print "%d mini batches" % (len(X_train[0]) / minibatch_size)

    best_dev_score = 0
    best_test_score = 0
    dev_scores = []
    test_scores = []

    for epoch in xrange(number_of_epochs):
        start_time = time.time()

        model.fit(X_train, Y_train, nb_epoch=1, batch_size=minibatch_size, verbose=1,
                  shuffle=True)

        print "%.2f sec for training" % (time.time() - start_time)

        pred_dev = model.predict(X_dev, verbose=0).argmax(axis=-1)  # Prediction of the classes
        dev_acc = np.sum(pred_dev == Y_dev) / float(len(Y_dev))
        dev_scores.append(dev_acc)
        pred_test = model.predict(X_test, verbose=0).argmax(axis=-1)  # test_case_x
        test_acc = np.sum(pred_test == Y_test) / float(len(Y_test))
        test_scores.append(test_acc)
        if dev_acc > best_dev_score:
            best_dev_score = dev_acc
            best_test_score = test_acc

        print "Accuracy dev: %.2f%% (best: %.2f%%)" % ((dev_acc * 100), best_dev_score * 100)
        print "Accuracy test: %.2f%% (best: %.2f%%)" % ((test_acc * 100), best_test_score * 100)

    print "POS Hidden weights sum (after POS train): ", np.sum(model.get_layer(name='pos_hidden').W.get_value())

    return dev_scores, test_scores
'''