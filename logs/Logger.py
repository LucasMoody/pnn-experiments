import datetime
def save_results(path, task, set, params, score, score_epoch):
    line = '{task}, {set}, {update_word_embeddings}, {window_size}, {batch_size}, {hidden_dims}, {activation},' \
           '{dropout}, {optimizer}, {score}, {score_epoch}, {datetime}\n'.format(
        task=task, set=set, update_word_embeddings=params['update_word_embeddings'], window_size=params['window_size'],
        batch_size=params['batch_size'], hidden_dims=params['hidden_dims'], activation=params['activation'],
        dropout=params['dropout'], optimizer=params['optimizer'], score=score, score_epoch=score_epoch,
        datetime=str(datetime.datetime.now())
    )
    with open(path, 'a') as f:
        f.write(line)


def save_reduced_datasets_results(path, experiment, task, set, params, score, score_epoch, dataset_size, add_models='none'):
    line = '{experiment}, {task}, {set}, {dataset_size}, {update_word_embeddings}, {window_size}, {batch_size}, {hidden_dims}, {activation},' \
           '{dropout}, {optimizer}, {score}, {score_epoch}, {datetime}, {add_models}\n'.format(
        task=task, set=set, update_word_embeddings=params['update_word_embeddings'], window_size=params['window_size'],
        batch_size=params['batch_size'], hidden_dims=params['hidden_dims'], activation=params['activation'],
        dropout=params['dropout'], optimizer=params['optimizer'], score=score, score_epoch=score_epoch,
        datetime=str(datetime.datetime.now()), dataset_size=dataset_size, experiment=experiment, add_models=add_models
    )
    with open(path, 'a') as f:
        f.write(line)

def save_scores_for_overfitting(train_scores, dev_scores):
    timeString = str(datetime.datetime.now())
    with open('overfitting.csv', 'a') as f:
        f.write(timeString + "\n")
        for i in xrange(len(train_scores)):
            f.write("{0:.2f}; {1:.2f}\n".format(train_scores[i], dev_scores[i]))


