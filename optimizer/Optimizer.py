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
        best_train_scores_pos, best_dev_scores_pos, best_test_scores_pos = OptimizedModels.getWSJPOSModel(params)
        print params
        for score in best_train_scores_pos:
            print "Max acc train pos: %.4f in epoch: %d" % (score[0], score[1])
            Logger.save_results(config.optimized_models_log_path, 'pos', 'train', params, score[0], score[1])
        for score in best_dev_scores_pos:
            print "Max acc dev pos: %.4f in epoch: %d" % (score[0], score[1])
            Logger.save_results(config.optimized_models_log_path, 'pos', 'dev', params, score[0], score[1])
        for score in best_test_scores_pos:
            print "Max acc test pos: %.4f in epoch: %d" % (score[0], score[1])
            Logger.save_results(config.optimized_models_log_path, 'pos', 'test', params, score[0], score[1])
    
        best_train_scores_ner, best_dev_scores_ner, best_test_scores_ner = OptimizedModels.getNERModel(params)
        print params
        for score in best_train_scores_ner:
            print "Max f1 train ner: %.4f in epoch: %d" % (score[0], score[1])
            Logger.save_results(config.optimized_models_log_path, 'ner', 'train', params, score[0], score[1])
        for score in best_dev_scores_ner:
            print "Max f1 dev ner: %.4f in epoch: %d" % (score[0], score[1])
            Logger.save_results(config.optimized_models_log_path, 'ner', 'dev', params, score[0], score[1])
        for score in best_test_scores_ner:
            print "Max f1 test ner: %.4f in epoch: %d" % (score[0], score[1])
            Logger.save_results(config.optimized_models_log_path, 'ner', 'test', params, score[0], score[1])
    
        best_train_scores_chunking, best_dev_scores_chunking, best_test_scores_chunking = OptimizedModels.getChunkingModel(params)
        print params
        for score in best_train_scores_chunking:
            print "Max f1 train chunking: %.4f in epoch: %d" % (score[0], score[1])
            Logger.save_results(config.optimized_models_log_path, 'chunking', 'train', params, score[0], score[1])
        for score in best_dev_scores_chunking:
            print "Max f1 dev chunking: %.4f in epoch: %d" % (score[0], score[1])
            Logger.save_results(config.optimized_models_log_path, 'chunking', 'dev', params, score[0], score[1])
        for score in best_test_scores_chunking:
            print "Max f1 test chunking: %.4f in epoch: %d" % (score[0], score[1])
            Logger.save_results(config.optimized_models_log_path, 'chunking', 'test', params, score[0], score[1])


def run_optimizer_with_fixed_params():
    fixed_params = {
        'update_word_embeddings': False,
        'window_size': 3,
        'batch_size': 128,
        'hidden_dims': 100,
        'activation': 'tanh',
        'dropout': 0.3,
        'optimizer': 'adam',
        'number_of_epochs': config.number_of_epochs
    }
    max_evals = config.number_of_evals

    for model_nr in xrange(max_evals):
        '''print "Model nr. ", model_nr
        print fixed_params
        best_train_scores_pos, best_dev_scores_pos, best_test_scores_pos = OptimizedModels.getWSJPOSModel(fixed_params)
        print fixed_params
        for score in best_train_scores_pos:
            print "Max acc train pos: %.4f in epoch: %d" % (score[0], score[1])
            Logger.save_results(config.optimized_models_log_path, 'pos', 'train', fixed_params, score[0], score[1])
        for score in best_dev_scores_pos:
            print "Max acc dev pos: %.4f in epoch: %d" % (score[0], score[1])
            Logger.save_results(config.optimized_models_log_path, 'pos', 'dev', fixed_params, score[0], score[1])
        for score in best_test_scores_pos:
            print "Max acc test pos: %.4f in epoch: %d" % (score[0], score[1])
            Logger.save_results(config.optimized_models_log_path, 'pos', 'test', fixed_params, score[0], score[1])

        best_train_scores_ner, best_dev_scores_ner, best_test_scores_ner = OptimizedModels.getNERModel(fixed_params)
        print fixed_params
        for score in best_train_scores_ner:
            print "Max f1 train ner: %.4f in epoch: %d" % (score[0], score[1])
            Logger.save_results(config.optimized_models_log_path, 'ner', 'train', fixed_params, score[0], score[1])
        for score in best_dev_scores_ner:
            print "Max f1 dev ner: %.4f in epoch: %d" % (score[0], score[1])
            Logger.save_results(config.optimized_models_log_path, 'ner', 'dev', fixed_params, score[0], score[1])
        for score in best_test_scores_ner:
            print "Max f1 test ner: %.4f in epoch: %d" % (score[0], score[1])
            Logger.save_results(config.optimized_models_log_path, 'ner', 'test', fixed_params, score[0], score[1])

        best_train_scores_chunking, best_dev_scores_chunking, best_test_scores_chunking = OptimizedModels.getChunkingModel(
            fixed_params)
        print fixed_params
        for score in best_train_scores_chunking:
            print "Max f1 train chunking: %.4f in epoch: %d" % (score[0], score[1])
            Logger.save_results(config.optimized_models_log_path, 'chunking', 'train', fixed_params, score[0], score[1])
        for score in best_dev_scores_chunking:
            print "Max f1 dev chunking: %.4f in epoch: %d" % (score[0], score[1])
            Logger.save_results(config.optimized_models_log_path, 'chunking', 'dev', fixed_params, score[0], score[1])
        for score in best_test_scores_chunking:
            print "Max f1 test chunking: %.4f in epoch: %d" % (score[0], score[1])
            Logger.save_results(config.optimized_models_log_path, 'chunking', 'test', fixed_params, score[0], score[1])'''

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
