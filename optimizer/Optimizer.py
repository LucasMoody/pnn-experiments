from optimizer import OptimizedModels
import random
from parameters import parameter_space
from logs import Logger
import config

max_evals = config.number_of_evals

for model_nr in xrange(max_evals):
    params = {}
    for key, values in parameter_space.space.iteritems():
        params[key] = random.choice(values)

    print "Model nr. ", model_nr
    model_ner, best_dev_scores, best_test_scores = OptimizedModels.getPOSModel(params)
    print params
    for score in best_dev_scores:
        print "Max acc dev: %.4f in epoch: %d" % (score[0], score[1])
        Logger.save_results(config.optimized_models_log_path, 'pos', 'dev', params, best_dev_scores)
    for score in best_test_scores:
        print "Max acc test: %.4f in epoch: %d" % (score[0], score[1])
        Logger.save_results(config.optimized_models_log_path, 'pos', 'test', params, best_test_scores)


