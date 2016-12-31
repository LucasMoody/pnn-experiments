from datasets.conll_ner import CoNLLNer
from embeddings.dependency_based_word_embeddings import DependencyBasedWordEmbeddings as Embeddings
from models import Trainer, InputBuilder
from models.NER import SennaNER as NER
from optimizer import OptimizedModels
from measurements import Measurer
import config
from parameters import parameter_space
from logs import Logger
import random

# settings
default_params = {
    'update_word_embeddings': False,
    'window_size': 3,
    'batch_size': 128,
    'hidden_dims': 100,
    'activation': 'tanh',
    'dropout': 0.3,
    'optimizer': 'adam'
}

number_of_epochs = config.number_of_epochs

'''windowSize = 3 # n to the left, n to the right
n_in = 2 * windowSize + 1
numHiddenUnitsPOS = 100
numHiddenUnitsNER = 100
n_minibatches = 1000
number_of_epochs = 1
metrics = []'''

# ----- metric results -----#

# Casing matrix
case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
            'contains_digit': 6, 'PADDING': 7}
n_in_case = len(case2Idx)

# Read in embeddings
embeddings = Embeddings.embeddings


def buildAndTrainNERModel(learning_params=None):
    if learning_params is None:
        params = default_params
    else:
        params = learning_params

    word2Idx = Embeddings.word2Idx
    [ner_input_train, ner_train_y_cat], [ner_input_dev, ner_dev_y], [ner_input_test,
                                                                     ner_test_y], ner_dicts = CoNLLNer.readDataset(
        params['window_size'], word2Idx, case2Idx)

    [ner_train_x, ner_train_case_x] = ner_input_train
    [ner_dev_x, ner_dev_case_x] = ner_input_dev
    [ner_test_x, ner_test_case_x] = ner_input_test
    [word2Idx, caseLookup, ner_label2Idx, ner_idx2Label] = ner_dicts
    ner_n_out = ner_train_y_cat.shape[1]

    model_train_input_ner = [ner_train_x, ner_train_case_x]
    model_dev_input_ner = [ner_dev_x, ner_dev_case_x]
    model_test_input_ner = [ner_test_x, ner_test_case_x]

    n_in_x = ner_train_x.shape[1]
    n_in_casing = ner_train_case_x.shape[1]

    input_layers_merged, inputs = InputBuilder.buildStandardModelInput(embeddings, case2Idx, n_in_x, n_in_casing)

    model_pos, _, _ = OptimizedModels.getPOSModelGivenInput(input_layers_merged, inputs, window_size=params['window_size'])

    model_ner = NER.buildNERModelWithPNN2(input_layers_merged, inputs, params, ner_n_out, additional_models=[model_pos])

    # ----- Train Model ----- #
    iof1 = Measurer.create_compute_IOf1(ner_idx2Label)
    dev_scores, test_scores = Trainer.trainModelWithIncreasingData(model_ner, model_train_input_ner,
                                                                   ner_train_y_cat, number_of_epochs,
                                                                   params['batch_size'], model_dev_input_ner,
                                                                   ner_dev_y, model_test_input_ner, ner_test_y,
                                                                   measurements=[iof1])

    return dev_scores, test_scores

max_evals = config.number_of_evals

for model_nr in xrange(max_evals):
    params = {}
    for key, values in parameter_space.space.iteritems():
        params[key] = random.choice(values)

    print "Model nr. ", model_nr
    best_dev_scores_ner, best_test_scores_ner = buildAndTrainNERModel(params)
    print params
    for (sample_scores, sample) in best_dev_scores_ner:
        for score in sample_scores:
            print "Max acc dev ner: %.4f in epoch with %d samples: %d" % (score[0][2], sample, score[1])
            Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_5', 'ner', 'dev', params, score[0][2], score[1], sample)
    for (sample_scores, sample) in best_test_scores_ner:
        for score in sample_scores:
            print "Max acc test ner: %.4f in epoch with %d samples: %d" % (score[0][2], sample, score[1])
            Logger.save_reduced_datasets_results(config.experiments_log_path, 'exp_5', 'ner', 'test', params, score[0][2], score[1], sample)