from logs import Logger
import config

def run_build_model(task, exp, params, build_model_func, score_name,
                    transfer_models='', transfer_config=None):
    if transfer_config is None:
        scores = build_model_func(params)
    else:
        scores = build_model_func(params, transfer_config)
        transfer_models = '-'.join(sorted(filter(lambda model: model not in ['words', 'casing', task], transfer_config)))

    print params
    for (train_score, dev_score, test_score, epoch, sample) in scores:
        print "Max {0} train/dev/test {1} with {2}: {3:.4f}/{4:.4f}/{5:.4f} in epoch: {6} with samples: {7}".format(
            score_name, task, transfer_models, train_score, dev_score, test_score, epoch, sample)
        Logger.save_reduced_datasets_results(
            config.experiments_log_path, exp, task, params,
            train_score, dev_score, test_score, epoch, sample, transfer_models)

    print '\n\n-------------------- END --------------------\n\n'