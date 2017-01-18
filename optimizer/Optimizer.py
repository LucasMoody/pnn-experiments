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
        best_train_scores_pos, best_dev_scores_pos, best_test_scores_pos = OptimizedModels.getPOSModel(params)
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
            print "Max f1 train ner: %.4f in epoch: %d" % (score[0][2], score[1])
            Logger.save_results(config.optimized_models_log_path, 'ner', 'train', params, score[0][2], score[1])
        for score in best_dev_scores_ner:
            print "Max f1 dev ner: %.4f in epoch: %d" % (score[0][2], score[1])
            Logger.save_results(config.optimized_models_log_path, 'ner', 'dev', params, score[0][2], score[1])
        for score in best_test_scores_ner:
            print "Max f1 test ner: %.4f in epoch: %d" % (score[0][2], score[1])
            Logger.save_results(config.optimized_models_log_path, 'ner', 'test', params, score[0][2], score[1])
    
        best_train_scores_chunking, best_dev_scores_chunking, best_test_scores_chunking = OptimizedModels.getChunkingModel(params)
        print params
        for score in best_train_scores_chunking:
            print "Max f1 train chunking: %.4f in epoch: %d" % (score[0][2], score[1])
            Logger.save_results(config.optimized_models_log_path, 'chunking', 'train', params, score[0][2], score[1])
        for score in best_dev_scores_chunking:
            print "Max f1 dev chunking: %.4f in epoch: %d" % (score[0][2], score[1])
            Logger.save_results(config.optimized_models_log_path, 'chunking', 'dev', params, score[0][2], score[1])
        for score in best_test_scores_chunking:
            print "Max f1 test chunking: %.4f in epoch: %d" % (score[0][2], score[1])
            Logger.save_results(config.optimized_models_log_path, 'chunking', 'test', params, score[0][2], score[1])


def run_optimizer_with_fixed_params():
    fixed_params = {
        'update_word_embeddings': False,
        'window_size': 3,
        'batch_size': 128,
        'hidden_dims': 100,
        'activation': 'relu',
        'dropout': 0.3,
        'optimizer': 'adam',
        'number_of_epochs': config.number_of_epochs
    }

    print fixed_params
    best_train_scores_pos, best_dev_scores_pos, best_test_scores_pos = OptimizedModels.getPOSModel(fixed_params)
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
        print "Max f1 train ner: %.4f in epoch: %d" % (score[0][2], score[1])
        Logger.save_results(config.optimized_models_log_path, 'ner', 'train', fixed_params, score[0][2], score[1])
    for score in best_dev_scores_ner:
        print "Max f1 dev ner: %.4f in epoch: %d" % (score[0][2], score[1])
        Logger.save_results(config.optimized_models_log_path, 'ner', 'dev', fixed_params, score[0][2], score[1])
    for score in best_test_scores_ner:
        print "Max f1 test ner: %.4f in epoch: %d" % (score[0][2], score[1])
        Logger.save_results(config.optimized_models_log_path, 'ner', 'test', fixed_params, score[0][2], score[1])

    best_train_scores_chunking, best_dev_scores_chunking, best_test_scores_chunking = OptimizedModels.getChunkingModel(
        fixed_params)
    print fixed_params
    for score in best_train_scores_chunking:
        print "Max f1 train chunking: %.4f in epoch: %d" % (score[0][2], score[1])
        Logger.save_results(config.optimized_models_log_path, 'chunking', 'train', fixed_params, score[0][2], score[1])
    for score in best_dev_scores_chunking:
        print "Max f1 dev chunking: %.4f in epoch: %d" % (score[0][2], score[1])
        Logger.save_results(config.optimized_models_log_path, 'chunking', 'dev', fixed_params, score[0][2], score[1])
    for score in best_test_scores_chunking:
        print "Max f1 test chunking: %.4f in epoch: %d" % (score[0][2], score[1])
        Logger.save_results(config.optimized_models_log_path, 'chunking', 'test', fixed_params, score[0][2], score[1])

run_optimizer_with_fixed_params()
