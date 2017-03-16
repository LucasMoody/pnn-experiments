import OptimizedModels
import random
from parameters import parameter_space
from logs import Logger
import config

def run_optimizer_with_random_params():
    
    max_evals = config.number_of_evals

    for model_nr in xrange(max_evals):
        params = {}
        for key, values in parameter_space.space.iteritems():
            params[key] = random.choice(values)
    
        print "Model nr. ", model_nr
        print params

        if 'ner' in config.tasks:
            run_build_model('ner', params, OptimizedModels.getNERModel, 'f1')

        if 'wsj_pos' in config.tasks:
            run_build_model('wsj_pos', params, OptimizedModels.getWSJPOSModel, 'acc')

        if 'ud_pos' in config.tasks:
            run_build_model('ud_pos', params, OptimizedModels.getUDPOSModel, 'acc')

        if 'chunking' in config.tasks:
            run_build_model('chunking', params, OptimizedModels.getChunkingModel, 'f1')


def run_optimizer_with_fixed_params():
    fixed_params = {
        'update_word_embeddings': False,
        'window_size': 3,
        'batch_size': 128,
        'hidden_dims': 100,
        'activation': 'tanh',
        'dropout': 0,
        'optimizer': 'adam',
        'number_of_epochs': config.number_of_epochs
    }
    max_evals = config.number_of_evals

    for model_nr in xrange(max_evals):
        print "Model nr. ", model_nr
        print fixed_params

        if 'ner' in config.tasks:
            run_build_model('ner', fixed_params, OptimizedModels.getNERModel, 'f1')

        if 'wsj_pos' in config.tasks:
            run_build_model('wsj_pos', fixed_params, OptimizedModels.getWSJPOSModel, 'acc')

        if 'ud_pos' in config.tasks:
            run_build_model('ud_pos', fixed_params, OptimizedModels.getUDPOSModel, 'acc')

        if 'chunking' in config.tasks:
            run_build_model('chunking', fixed_params, OptimizedModels.getChunkingModel, 'f1')

def run_build_model(task, params, build_model_func, score_name):
    train_scores, dev_scores, test_scores = build_model_func(params)
    print params
    for (score, epoch) in train_scores:
        print "Max {0} train {1}: {2:.4f} in epoch: {3}".format(score_name, task, score, epoch)
        Logger.save_results(config.optimized_models_log_path, task, 'train', params, score, epoch)
    for (score, sample) in dev_scores:
        print "Max {0} dev {1}: {2:.4f} in epoch: {3}".format(score_name, task, score, epoch)
        Logger.save_results(config.optimized_models_log_path, task, 'dev', params, score, epoch)
    for (score, sample) in test_scores:
        print "Max {0} test {1}: {2:.4f} in epoch: {3}".format(score_name, task, score, epoch)
        Logger.save_results(config.optimized_models_log_path, task, 'test', params, score, epoch)

    print '\n\n-------------------- END --------------------\n\n'

run_optimizer_with_fixed_params()
