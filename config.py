import os

optimized_models_log_path = os.environ.get('OPTIMIZED_MODELS_LOG_PATH')
experiments_log_path = os.environ.get('EXPERIMENTS_LOG_PATH')
number_of_epochs = int(os.environ.get('NUMBER_OF_EPOCHS'))
number_of_evals = int(os.environ.get('NUMBER_OF_EVALS'))
number_of_samples = int(os.environ.get('NUMBER_OF_SAMPLES'))
tasks = os.environ.get('TASKS', 'ner,chunking,pos').split(',')
dev_env = os.environ.get('DEV', 'False') == 'True'
patience = int(os.environ.get('PATIENCE', 10))
samples = os.environ.get('SAMPLES', '').split(',')
ed_source_tasks = os.environ.get('ED_SOURCE_TASKS', 'uncategorized,categorized').split(',')
adapter_size = int(os.environ.get('ADAPTER_SIZE', 10))