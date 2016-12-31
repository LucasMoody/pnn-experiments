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


def save_reduced_datasets_results(path, experiment, task, set, params, score, score_epoch, dataset_size):
    line = '{experiment}, {task}, {set}, {dataset_size}, {update_word_embeddings}, {window_size}, {batch_size}, {hidden_dims}, {activation},' \
           '{dropout}, {optimizer}, {score}, {score_epoch}, {datetime}\n'.format(
        task=task, set=set, update_word_embeddings=params['update_word_embeddings'], window_size=params['window_size'],
        batch_size=params['batch_size'], hidden_dims=params['hidden_dims'], activation=params['activation'],
        dropout=params['dropout'], optimizer=params['optimizer'], score=score, score_epoch=score_epoch,
        datetime=str(datetime.datetime.now()), dataset_size=dataset_size, experiment=experiment
    )
    with open(path, 'a') as f:
        f.write(line)