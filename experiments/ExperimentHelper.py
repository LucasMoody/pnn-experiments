from logs import Logger
import config

def run_build_model(task, exp, params, build_model_func, score_name,
                    transfer_models, transfer_config):
    train_scores, dev_scores, test_scores = build_model_func(params, transfer_config)
    print params
    for (sample_scores, sample) in train_scores:
        for score in sample_scores:
            print "Max {0} train {1} with {2}: {3:.4f} in epoch: {4} with samples: {5}".format(
                score_name, task, transfer_models, score[0], score[1], sample)
            Logger.save_reduced_datasets_results(
                config.experiments_log_path, exp, task, 'train', params,
                score[0], score[1], sample, transfer_models)
    for (sample_scores, sample) in dev_scores:
        for score in sample_scores:
            print "Max {0} dev {1} with {2}: {3:.4f} in epoch: {4} with samples: {5}".format(
                score_name, task, transfer_models, score[0], score[1], sample)
            Logger.save_reduced_datasets_results(
                config.experiments_log_path, exp, task, 'dev', params,
                score[0], score[1], sample, transfer_models)
    for (sample_scores, sample) in test_scores:
        for score in sample_scores:
            print "Max {0} test {1} with {2}: {3:.4f} in epoch: {4} with samples: {5}".format(
                score_name, task, transfer_models, score[0], score[1], sample)
            Logger.save_reduced_datasets_results(
                config.experiments_log_path, exp, task, 'test', params,
                score[0], score[1], sample, transfer_models)

    print '\n\n-------------------- END --------------------\n\n'