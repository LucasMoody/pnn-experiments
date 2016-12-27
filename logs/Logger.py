import datetime
def save_results(path, task, set, params, scores):
    line = '{task}, {set}, {update_word_embeddings}, {window_size}, {batch_size}, {hidden_dims}, {activation},' \
           '{dropout:}, {optimizer}, {score}, {score_epoch}, {datetime}\n'.format(
        task=task, set=set, update_word_embeddings=params['update_word_embeddings'], window_size=params['window_size'],
        batch_size=params['batch_size'], hidden_dims=params['hidden_dims'], activation=params['activation'],
        dropout=params['dropout'], optimizer=params['optimizer'], score=scores[0][0], score_epoch=scores[0][1],
        datetime=str(datetime.datetime.now())
    )
    with open(path, 'a') as f:
        f.write(line)