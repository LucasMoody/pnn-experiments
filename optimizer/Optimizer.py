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
        'batch_size': 16,
        'hidden_dims': 100,
        'activation': 'tanh',
        'dropout': 0.3,
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

        if 'ace' in config.tasks:
            run_build_model('ace', fixed_params, OptimizedModels.getAceEDModel, 'f1')

        if 'ace_wo_contacts' in config.tasks:
            run_build_model('ace_wo_contacts', fixed_params, OptimizedModels.getAceWoContactsEDModel, 'f1')

        if 'ace_wo_business' in config.tasks:
            run_build_model('ace_wo_business', fixed_params, OptimizedModels.getAceWoBusinessEDModel, 'f1')

        if 'ace_wo_justice' in config.tasks:
            run_build_model('ace_wo_justice', fixed_params, OptimizedModels.getAceWoJusticeEDModel, 'f1')

        if 'ace_wo_movement' in config.tasks:
            run_build_model('ace_wo_movement', fixed_params, OptimizedModels.getAceWoMovementEDModel, 'f1')

        if 'tac' in config.tasks:
            run_build_model('tac', fixed_params, OptimizedModels.getTacEDModel, 'f1')

        if 'tempeval' in config.tasks:
            run_build_model('tempeval', fixed_params, OptimizedModels.getTempevalEDModel, 'f1')

        if 'ecb' in config.tasks:
            run_build_model('ecb', fixed_params, OptimizedModels.getEcbPlusEDModel, 'f1')

def run_build_model(task, params, build_model_func, score_name):
    best_train_score, best_dev_score, best_test_score, best_epoch = build_model_func(params)
    print params
    print "Max {0} train/dev/test {1}: {2:.4f}/{3:.4f}/{4:.4f} in epoch: {5}".format(score_name, task, best_train_score, best_dev_score, best_test_score, best_epoch)
    Logger.save_results(config.optimized_models_log_path, task, params, best_train_score, best_dev_score, best_test_score, best_epoch)
    print '\n\n-------------------- END --------------------\n\n'

run_optimizer_with_fixed_params()
