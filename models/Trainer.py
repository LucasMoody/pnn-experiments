import time
import numpy as np
import Sampler
import config
import copy
from plots import LearningCurve
from logs import Logger
#sample_fun = Sampler.sampleEqualRanges
#sample_fun = Sampler.sampleLog2Ranges
#sample_fun = Sampler.sampleLog2AndEqualRanges
sample_fun = Sampler.sampleSimplePNNRanges

no_samples = config.number_of_samples
early_stopping_strike = 10 if not config.dev_env else 3

def trainModel(model, X_train, Y_train, number_of_epochs, minibatch_size, X_dev, Y_dev, X_test, Y_test, measurer):
    print "%d epochs" % number_of_epochs
    print "%d mini batches" % (len(X_train[0]) / minibatch_size)

    best_epoch = 0
    best_train_score = -1
    best_dev_score = -1
    best_test_score = -1
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
        start_test_time = time.time()
        # only dev scores need to be calculated
        pred_dev = model.predict(X_dev, verbose=0).argmax(axis=-1)  # Prediction of the classes
        score_dev = measurer(pred_dev, Y_dev)
        #pred_train = model.predict(X_train, verbose=0).argmax(axis=-1)  # Prediction of the classes
        #measurements_train = measurer(pred_train, Y_train.argmax(axis=1))
        # update best scores
        # compare dev scores to get best one
        if score_dev > best_dev_score:
            best_dev_score = score_dev
            best_epoch = epoch
            best_model_weights = map(lambda x: x.copy(), model.get_weights())

        print "%.2f sec for testing" % (time.time() - start_test_time)
        print 'Current dev_score: {0:.4f} and current patience: {1}'.format(score_dev * 100, epoch - best_epoch)
        #print 'Current train_score/dev_score: {0:.4f}/{1:.4f} and current patience: {2}'.format(measurements_train[0] * 100, measurements_dev[0] * 100, epoch - best_dev_scores[0][1])
        #dev_scores.append(measurements_dev[0] * 100)
        #train_scores.append(measurements_train[0] * 100)
        # early stopping
        if epoch - best_epoch > early_stopping_strike:
            break



    #Logger.save_scores_for_overfitting(train_scores, dev_scores)

    # ----- score calculations and weight resetting to best score ----- #
    # set back weights to best epoch
    print 'Weight sum before setting best epoch:', reduce(lambda a, b: a + np.sum(b), model.get_weights(), 0)
    model.set_weights(best_model_weights)
    print 'Weight sum after finished training:', reduce(lambda a, b: a + np.sum(b), model.get_weights(), 0)

    # calculate train and test scores for best epoch as well
    # make predictions on other datasets
    # if not train set is not the full train dataset
    pred_train = model.predict(X_train, verbose=0).argmax(axis=-1)  # Prediction of the classes
    pred_dev = model.predict(X_dev, verbose=0).argmax(axis=-1)  # Prediction of the classes
    pred_test = model.predict(X_test, verbose=0).argmax(axis=-1)  # test_case_x
    # calculate scores of predictions
    best_train_score = measurer(pred_train, Y_train.argmax(axis=1))
    score_dev = measurer(pred_dev, Y_dev)
    best_test_score = measurer(pred_test, Y_test)

    # test whether earlier calculated dev score is the same as with reset weights
    if best_dev_score != score_dev:
        raise ValueError('Newly calculated best score should be the same as earlier saved one!')
    print 'best train/dev/test score: {0:.4f}/{1:.4f}/{2:.4f} in epoch: {3}'.format(best_train_score * 100,
                                                                                    best_dev_score * 100,
                                                                                    best_test_score * 100,
                                                                                    best_epoch)
    return best_train_score, best_dev_score, best_test_score, best_epoch

def trainModelWithIncreasingData(model, X_train, Y_train, number_of_epochs, minibatch_size, X_dev, Y_dev, X_test, Y_test, measurer):
    ranges = sample_fun(X_train, no_samples)

    scores = []

    print "%d samples" % no_samples
    weights = map(lambda x: x.copy(), model.get_weights())

    for sample in ranges:
        print "Current sample: 0:%d" % sample
        start_time = time.time()

        sampled_train_x = map(lambda x: x[0:sample], X_train)
        sampled_train_y = Y_train[0:sample]

        print 'Weight sum before training', reduce(lambda a, b: a + np.sum(b), model.get_weights(), 0)
        best_train_score, best_dev_score, best_test_score, epoch = trainModel(model, sampled_train_x, sampled_train_y, number_of_epochs, minibatch_size, X_dev, Y_dev, X_test, Y_test, measurer)
        model.set_weights(weights)
        print 'Weight sum after resetting', reduce(lambda a, b: a + np.sum(b), model.get_weights(), 0)
        print "%.2f sec for sample training" % (time.time() - start_time)
        scores.append([best_train_score, best_dev_score, best_test_score, epoch, sample])

    return scores

def trainMultiTaskModelWithIncreasingData(models, datasets, number_of_epochs, minibatch_size):
    focused_dataset = datasets[0]
    ranges = sample_fun(focused_dataset['train']['input'], no_samples)

    scores = []

    print "%d samples" % no_samples
    model_weights = map(lambda model: map(lambda x: x.copy(), model.get_weights()), models)

    for sample in ranges:
        print "Current sample: 0:%d" % sample
        start_time = time.time()
        # replace train for focused dataset
        # copy everything because datasets are going to be shuffled
        def copyDataset(dataset):
            return {
                'train': {
                    'input': list(map(lambda input: input.copy(), dataset['train']['input'].copy())),
                    'y': dataset['train']['y'].copy()
                },
                'dev': {
                    'input': list(map(lambda input: input.copy(), dataset['dev']['input'].copy())),
                    'y': dataset['dev']['y'].copy()
                },
                'test': {
                    'input': list(map(lambda input: input.copy(), dataset['test']['input'].copy())),
                    'y': dataset['test']['y'].copy()
                }
            }
        #cur_datasets = list(map(lambda data: copyDataset(data), datasets))
        cur_datasets = copy.deepcopy(datasets)
        cur_datasets[0]['train']['input'] = map(lambda x: x[0:sample], datasets[0]['train']['input'])
        cur_datasets[0]['train']['y'] = cur_datasets[0]['train']['y'][0:sample]

        print 'Weight sum before training', sum_model_weights(models)
        best_train_score, best_dev_score, best_test_score, epoch = trainMultiTaskModels(models, cur_datasets, number_of_epochs, minibatch_size)
        for idx, model in enumerate(models):
            weights = model_weights[idx]
            model.set_weights(weights)
        print 'Weight sum after resetting', sum_model_weights(models)
        print "%.2f sec for sample training" % (time.time() - start_time)
        scores.append([best_train_score, best_dev_score, best_test_score, epoch, sample])

    return scores

def trainMultiTaskModels(models, datasets, number_of_epochs, minibatch_size):
    best_epoch = 0
    best_train_score = -1
    best_dev_score = -1
    best_test_score = -1

    best_model_weights = map(lambda model: map(lambda x: x.copy(), model.get_weights()), models)
    # maximum number of epochs
    global_batch_start = 0
    virtual_epoch_counter = 0
    while virtual_epoch_counter < number_of_epochs:
        start_time = time.time()
        # models are trained by turns
        for idx, model in enumerate(models):
            model_data = datasets[idx]
            input_data = model_data['train']['input']
            train_y = model_data['train']['y']
            input_size = input_data[0].shape[0]
            current_batch_start = global_batch_start % input_size
            current_batch_end = current_batch_start + minibatch_size if current_batch_start + minibatch_size < input_size else input_size

            model.train_on_batch(map(lambda x: x[current_batch_start:current_batch_end],input_data), train_y[current_batch_start:current_batch_end])
            # check whether current dataset has to be shuffled
            if current_batch_end == input_size:
                input, y = unison_shuffled_copies(model_data['train']['input'], model_data['train']['y'])
                model_data['train']['input'] = input
                model_data['train']['y'] = y

        # measuring after all samples of the focused model were trained
        focused_model_data = datasets[0]
        focused_model_data_size = focused_model_data['train']['input'][0].shape[0]
        global_batch_start += minibatch_size
        if (global_batch_start % focused_model_data_size) - minibatch_size < 0:
            print "%.2f sec for training" % (time.time() - start_time)
            start_prediction_time = time.time()
            dev_scores = []
            # calculate the scores for each model in dev mode or only the target model score in other modes
            for idx, model in (enumerate(models) if config.dev_env else enumerate([models[0]])):
                model_data = datasets[idx]
                dev_input = model_data['dev']['input']
                pred_dev = model.predict(dev_input, verbose=0).argmax(axis=-1)  # Prediction of the classes
                measurements_dev = model_data['measurer'](pred_dev, model_data['dev']['y'])
                print 'Model for {0} with score: {1:.4f}'.format(model_data['name'], measurements_dev * 100)
                dev_scores.append(measurements_dev)
            # check whether the best score of the focused model is improved
            if dev_scores[0] > best_dev_score:
                best_dev_score = dev_scores[0]
                best_epoch = virtual_epoch_counter
                best_model_weights = map(lambda model: map(lambda x: x.copy(), model.get_weights()), models)
                print 'Weight sum in best epoch:', sum_model_weights(models)
            # early stopping check
            patience = virtual_epoch_counter - best_epoch
            print '{0:.4f}/{1:.4f} with patience: {2}'.format(best_dev_score, dev_scores[0], patience)
            if patience > early_stopping_strike:
                break
            virtual_epoch_counter += 1
            print "%.2f sec for prediction" % (time.time() - start_prediction_time)

    # ----- score calculations and weight resetting to best score ----- #
    # set back weights to best epoch
    print 'Weight sum before setting best epoch:', sum_model_weights(models)
    for idx, model in enumerate(models):
        cur_model_weights = best_model_weights[idx]
        model.set_weights(cur_model_weights)
    print 'Weight sum after finished training:', sum_model_weights(models)

    # make predictions on other datasets
    # if not train set is not the full train dataset
    focused_model = models[0]
    focused_train_x = datasets[0]['train']['input']
    focused_dev_x = datasets[0]['dev']['input']
    focused_test_x = datasets[0]['test']['input']
    focused_train_y = datasets[0]['train']['y']
    focused_dev_y = datasets[0]['dev']['y']
    focused_test_y = datasets[0]['test']['y']
    pred_train = focused_model.predict(focused_train_x, verbose=0).argmax(axis=-1)  # Prediction of the classes
    pred_dev = focused_model.predict(focused_dev_x, verbose=0).argmax(axis=-1)  # Prediction of the classes
    pred_test = focused_model.predict(focused_test_x, verbose=0).argmax(axis=-1)  # test_case_x
    # calculate scores of predictions
    best_train_score = datasets[0]['measurer'](pred_train, focused_train_y.argmax(axis=1))
    score_dev = datasets[0]['measurer'](pred_dev, focused_dev_y)
    best_test_score = datasets[0]['measurer'](pred_test, focused_test_y)

    # test whether earlier calculated dev score is the same as with reset weights
    if best_dev_score != score_dev:
        raise ValueError('Newly calculated best score should be the same as earlier saved one!')

    print 'best train/dev/test score: {0:.4f}/{1:.4f}/{2:.4f} in epoch: {3}'.format(best_train_score * 100, best_dev_score * 100, best_test_score * 100, best_epoch)
    return best_train_score, best_dev_score, best_test_score, best_epoch

def unison_shuffled_copies(input, y):
    for x in input:
        assert len(x) == len(y)
    p = np.random.permutation(len(y))
    return map(lambda x: x[p], input), y[p]

def sum_model_weights(models):
    return reduce(lambda total, cur_sum: total + cur_sum,
           map(lambda model: reduce(lambda a, b: a + np.sum(b), model.get_weights(), 0), models), 0)