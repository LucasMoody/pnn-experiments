#Transfer Learning with Progressive Neural Networks for Event Detection

## Requirements 
* To run the code, you need Python 2.7, Theano (tested on Theano 0.8.2) and Keras (tested on 1.1.2)
* Levy's word embeddings are required (we used the word embeddings based on Dependency links). You should put the file in the `embeddings/dependency_based_word_embeddings/vocabs/` folder and name it `levy_dependency_based.words.vocab`
* The `levy_dependency_based.words.vocab` should include a 'UNKNOWN' token for out of vacabulary tokens
* You need the following datasets:
  * ACE event detection dataset. Put the `train.txt`, `dev.txt` and `test.text` set into the `datasets/ace_ed/data` folder
  * TAC event detection dataset. Put the `train.txt`, `dev.txt` and `test.text` set into the `datasets/tac2015_ed/data` folder
  * ECB+ event detection dataset. Put the `train.txt`, `dev.txt` and `test.text` set into the `datasets/ecbplus_ed/data` folder
  * TempEval event detection dataset. Put the `train.txt`, `dev.txt` and `test.text` set into the `datasets/tempeval3_ed/data` folder
  * Universal Dependencies POS dataset. Put the `en-ud-train.conllu`, `en-ud-dev.conllu` and `en-ud-test.conllu` set into the `datasets/universal_dependencies_pos/data` folder
  * PennTreeBank POS dataset. Put the `train.txt`, `dev.txt` and `test.text` set into the `datasets/wsj_pos/data` folder
  * CoNLL NER dataset. Put the `train.txt`, `dev.txt` and `test.text` set into the `datasets/conll_ner/data` folder
  * CoNLL Chunking dataset. Put the `train.txt`, `dev.txt` and `test.text` set into the `datasets/conll_chunking/data` folder

## Project Structure
```
PNN
│───datasets : All datasets and readers and reader helpers are located here
│   └─ace_ed
│       └─ACEED.py : Includes readDataset and readDatasetExtended (with features for pipeline) methods
│   └─ ...
│
└───embeddings: contains levy embeddings and an embedding class
│
└───experiments: contains experiments to test the hypotheses
│   └─eventdetection: contains event detection experiments
│   │   └─baseline.py : Runs the baseline model for the given tasks (ACE, TAC, ECB, TempEval)
│   │   └─finetuning.py : Runs the finetuning model for the given tasks
│   │   └─multitask.py : Runs the multitask model for the given tasks
│   │   └─pipeline.py : Runs the pipeline model for the given tasks
│   │   └─pnn_ext.py : Runs the extended PNN model for the given tasks
│   │   └─pnn_ext_adapter.py : Runs the PNN extended with an adapter model for the given tasks
│   │   └─pnn_original.py : Runs the original PNN model for the given tasks
│   │   └─pnn_original_adapter.py : Runs the original PNN with an adapter model for the given tasks
│   │
│   └─standardnlp : contains experiments for the three standard NLP tasks (the same models as in event detection)
│       └─ ...
│   
└───logs : contains the logging functions
│
└───measurements : contains the measurer functions like BIO-F1 and accuracy
│
└───models : contains Keras models and the training functions
│   └─InputBuilder.py : Helper functions for building the input layers
│   └─Sampler.py : Sampler functions but actually only sampleSimplePNNRanges was used
│   └─Senna.py : Includes functions for the baseline, fine-tuning, pipeline, multi-task and the different PNN models
│   └─Trainer.py : Includes functions for training a model for a single sample size, multiple sizes and the same for the multi-task learning model
│
└───optimizer : Includes scripts for training the source models
│   └─saved_models : Includes the weight files of the optimized models in the hd5 format 
│   └─OptimizedModels.py : Functions for training and getting optimized models for a task
│   └─Optimizer.py : Functions for training the optimized models for different tasks
│
└───parameters : Includes a config file which contains all possible hyperparameters
│
└───plots : Includes a function for plotting a learning curve (was not used in thesis)
│
└───statistics : Includes functions for calculating the results for this thesis and descriptive statistics
│
└───transfer : Helper functions for transfer learning (includes functions for Frustrantingly Easy Domain Adaptation and Return of... => was not used)
│
└───config.py : Loads all the environment variables which are used for configuration
```

## Executing the scripts
The following environments variables can or must be set:
* KERAS_BACKEND=theano
* OMP_NUM_THREADS=X : X number of threads are used by theano
* THEANO_FLAGS=device=[cpu,gpu],floatX=float32 : theano used cpu or gpu
* NUMBER_OF_EVALS=X : The script performs X number of evaluations of one experiments
* NUMBER_OF_EPOCHS=X : The Trainer.py script trains a model for maximum X epochs (MAXIMUM_NO_OF_EPOCHS would be better). Early stopping is used, so you can choose a high number.
* OPTIMIZED_MODELS_LOG_PATH=X : The Optimizer script saves the results in file X
* EXPERIMENTS_LOG_PATH=X : The results of the experiments are saved in csv format in file X
* NUMBER_OF_SAMPLES=X (default is '' meaning no samples): The sampleEqualRanges of the Sampler script creates X number of samples for training (not used in this thesis)
* TASKS=X,Y,Z,... (default is ace) : The default task ace is performed. The following tasks can be used in the experiments:
  * ace
  * ace_only_contacts
  * ace_only_justice
  * ace_only_business (not used in this thesis)
  * ace_only_movement (not used in this thesis)
  * tac
  * tac_newswire
  * tac_forum
  * ecb
  * tempeval
  * ner
  * chunking
  * wsj_pos (PennTreeBank POS)
  * ud_pos
* PATIENCE=X (default is 10) : Early stopping is used with X patience
* SAMPLES=X,Y,Z,[ALL] : The Trainer script performs experiments for the sample sizes X, Y, Z and for the full dataset when ALL is mentioned. The default is no sample size.
* ADAPTER_SIZE=X (default is 10): The size of the adapter in the original and extended PNN architecture with an adpater. The default is 10
* PYTHONPATH=X : Must be mentioned because the scripts are not located in the current path. Just use $(pwd) when ure in the pnn directory

Here are some example executions:
```
KERAS_BACKEND=theano OMP_NUM_THREADS=4 THEANO_FLAGS=device=cpu,floatX=float32  NUMBER_OF_EVALS=10 NUMBER_OF_EPOCHS=1000 OPTIMIZED_MODELS_LOG_PATH=~/optimized_results.csv EXPERIMENTS_LOG_PATH=~/experiments_results.csv TASKS=ace,ecb,tac,tempeval SAMPLES=2000,5000,10000,15000,20000,30000,ALL ADAPTER_SIZE=100  PYTHONPATH=$(pwd) python experiments/eventdetection/pnn_ext_adapter.py
```
This performs 10 evaluations of an experiment with the PNN extended with an adapter of size 100. The script performs the experiment for 2,000, 5,000, 10,000, 15,000, 20,000, 30,000 and all examples. Early stopping with a patience of 10 is used. Results are saved in the experiments_results.csv file in the user home.

```
KERAS_BACKEND=theano OMP_NUM_THREADS=4 THEANO_FLAGS=device=cpu,floatX=float32  NUMBER_OF_EVALS=30 NUMBER_OF_EPOCHS=1000 OPTIMIZED_MODELS_LOG_PATH=~/optimized_results.csv EXPERIMENTS_LOG_PATH=~/experiments_results.csv TASKS=tac_newswire SAMPLES=ALL PYTHONPATH=$(pwd) python experiments/eventdetection/baseline.py
```
This performs an experiment with the baseline model on the tac newswire task.

```
KERAS_BACKEND=theano OMP_NUM_THREADS=4 THEANO_FLAGS=device=cpu,floatX=float32  NUMBER_OF_EVALS=10 NUMBER_OF_EPOCHS=1000 OPTIMIZED_MODELS_LOG_PATH=~/optimized_results.csv EXPERIMENTS_LOG_PATH=~/experiments_results.csv TASKS=wsj_pos,ner,chunking SAMPLES=2000,5000,10000,15000,20000,30000,ALL PYTHONPATH=$(pwd) python experiments/standardnlp/pipeline.py
```
This performs an experiment with the pipeline model on the NER, Chunking and PennTreeBank POS task.

```
KERAS_BACKEND=theano OMP_NUM_THREADS=4 THEANO_FLAGS=device=cpu,floatX=float32  NUMBER_OF_EVALS=10 NUMBER_OF_EPOCHS=1000 OPTIMIZED_MODELS_LOG_PATH=~/optimized_results.csv EXPERIMENTS_LOG_PATH=~/experiments_results.csv TASKS=ace,ace_without_contacts,ace_without_justice,tac_newswire,tac_forum,ecb,tempeval,ner,wsj_pos,ud_pos,chunking PYTHONPATH=$(pwd) python optimizer/Optimizer.py
```
The optimizer trains optimized models for all source tasks which are used in this thesis. The weights are saved in optimizer/saved_models/ with score on dev and test set in the name.