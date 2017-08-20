import os

optimized_models_log_path = os.environ.get('OPTIMIZED_MODELS_LOG_PATH')
experiments_log_path = os.environ.get('EXPERIMENTS_LOG_PATH')
number_of_epochs = int(os.environ.get('NUMBER_OF_EPOCHS'))
number_of_evals = int(os.environ.get('NUMBER_OF_EVALS'))
number_of_samples = int(os.environ.get('NUMBER_OF_SAMPLES'))
tasks = os.environ.get('TASKS', 'ner,chunking,pos').split(',')
dev_env = os.environ.get('DEV', 'False') == 'True'
patience = int(os.environ.get('PATIENCE', 10))
