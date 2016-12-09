from embeddings.dependency_based_word_embeddings import DependencyBasedWordEmbeddings as Embeddings
from models import OptimizedModels
from models.NER import SennaNER as NER
from datasets.conll_ner import CoNLLNer
from keras.layers import Input, Embedding, Flatten, merge
from models import Trainer
import plots.LearningCurve as LearningCurve

# settings
windowSize = 3 # n to the left, n to the right
n_in = 2 * windowSize + 1
numHiddenUnitsPOS = 100
numHiddenUnitsNER = 100
n_minibatches = 1000
number_of_epochs = 1
metrics = []

# ----- metric results -----#
metric_results = []

#Casing matrix
caseLookup = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING':7}
n_in_case = len(caseLookup)

# Read in embeddings
embeddings = Embeddings.embeddings
word2Idx = Embeddings.word2Idx

def buildAndTrainNERModel():
    (ner_train_x, ner_train_case_x, ner_train_y, ner_train_y_cat), (ner_dev_x, ner_dev_case_x, ner_dev_y), (
        ner_test_x, ner_test_case_x, ner_test_y) = CoNLLNer.readDataset(windowSize, word2Idx, caseLookup)

    input_train = [ner_train_x, ner_train_case_x]
    input_dev = [ner_dev_x, ner_dev_case_x]
    input_test = [ner_test_x, ner_test_case_x]

    ner_n_out = ner_train_y_cat.shape[1]
    n_in_x = ner_train_x.shape[1]
    n_in_casing = ner_train_case_x.shape[1]

    model_train_input_ner = [ner_train_x, ner_train_case_x]
    model_dev_input_ner = [ner_dev_x, ner_dev_case_x]
    model_test_input_ner = [ner_test_x, ner_test_case_x]

    pos_model = OptimizedModels.getPOSModel(embeddings, word2Idx)

    words_input = Input(shape=(n_in_x,), dtype='int32', name='words_input')
    wordEmbeddingLayer = Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in_x,
                                   weights=[embeddings], trainable=False)
    words = wordEmbeddingLayer(words_input)
    words = Flatten(name='words_flatten')(words)

    case_input = Input(shape=(n_in_x,), dtype='int32', name='case_input')
    caseEmbeddingLayer = Embedding(output_dim=len(caseLookup), input_dim=len(caseLookup), input_length=n_in_casing,
                                   trainable=True)
    casing = caseEmbeddingLayer(case_input)
    casing = Flatten(name='casing_flatten')(casing)

    input_layers = [words, casing]
    inputs = [words_input, case_input]

    input_layers_merged = merge(input_layers, mode='concat')

    model_ner = NER.buildNERModelGivenInput(input_layers_merged, inputs, numHiddenUnitsNER, ner_n_out, useHiddenWeights=True, additional_models=[pos_model])

    dev_accs_ner, test_accs_ner, dev_f1s_ner, test_f1s_ner, ranges = Trainer.trainModelWithIncreasingData(model_ner,
                                                                                                  input_train,
                                                                                                  ner_train_y_cat,
                                                                                                  number_of_epochs,
                                                                                                  n_minibatches,
                                                                                                  input_dev, ner_dev_y,
                                                                                                  input_test,
                                                                                                  ner_test_y)

    return dev_accs_ner, test_accs_ner, dev_f1s_ner, test_f1s_ner, ranges


'''dev_accs_ner, test_accs_ner, dev_f1s_ner, test_f1s_ner = buildAndTrainNERModel()

metric_results.append((dev_accs_ner, 'ner_dev_acc'))
metric_results.append((test_accs_ner, 'ner_test_acc'))
metric_results.append((dev_f1s_ner, 'ner_dev_f1'))
metric_results.append((test_f1s_ner, 'ner_test_f1'))

LearningCurve.plotLearningCurve(metric_results)'''