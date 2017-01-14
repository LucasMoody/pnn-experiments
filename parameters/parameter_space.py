import config

space = {
    'update_word_embeddings': [False],#, True],
    'window_size': range(1, 4 + 1),
    'batch_size': [32,64,128],
    'hidden_dims': range(20,300+1,5),
    'activation':  ['relu', 'tanh', 'sigmoid'],
    'dropout': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75 ],
    'optimizer': ['adagrad', 'adadelta','adam','adamax','rmsprop', 'nadam'],
    'number_of_epochs': [config.number_of_epochs]
}