from datasets.universal_dependencies_pos import UDPos
from datasets.conll_ner import CoNLLNer
import embeddings.dependency_based_word_embeddings.DependencyBasedWordEmbeddings as Embeddings
from transfer import Extender
import numpy as np
import models.POS.SennaPOS as POS
import models.NER.SennaNER as NER
import models.Trainer as Trainer
from models import OptimizedModels
from keras.layers import Input, Embedding, Flatten, merge
from keras.models import Model
import theano
from models.POS import SennaPOS as POS
from models.NER import SennaNER as NER

from keras.utils import np_utils
from measurements import Measurer
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

def extendCoNLLNer():
    (ner_train_x, ner_train_case_x, ner_train_y, ner_train_y_cat), (ner_dev_x, ner_dev_case_x, ner_dev_y), (
            ner_test_x, ner_test_case_x, ner_test_y) = CoNLLNer.readDataset(windowSize, word2Idx, caseLookup)

    model_train_input_ner = [ner_train_x, ner_train_case_x]
    model_dev_input_ner = [ner_dev_x, ner_dev_case_x]
    model_test_input_ner = [ner_test_x, ner_test_case_x]

    pos_model = OptimizedModels.getPOSModel(embeddings, word2Idx)
    pred_train = pos_model.predict(model_train_input_ner, verbose=0).argmax(axis=-1)
    pred_dev = pos_model.predict(model_dev_input_ner, verbose=0).argmax(axis=-1)
    pred_test = pos_model.predict(model_test_input_ner, verbose=0).argmax(axis=-1)

    pos_label2Idx, pos_idx2Label = UDPos.getLabelDict()
    pred_train_labels = map(lambda idx: pos_idx2Label[idx], pred_train)
    pred_dev_labels = map(lambda idx: pos_idx2Label[idx], pred_dev)
    pred_test_labels = map(lambda idx: pos_idx2Label[idx], pred_test)

    train_extensions = [pred_train_labels]
    dev_extensions = [pred_dev_labels]
    test_extensions = [pred_test_labels]

    CoNLLNer.extendDataset("./datasets/conll_ner/data/eng.conllu", train_extensions, dev_extensions, test_extensions)

def extendUDPOS():
    # Read in files
    (pos_train_x, pos_train_case_x, pos_train_y, pos_train_y_cat), (pos_dev_x, pos_dev_case_x, pos_dev_y), (
        pos_test_x, pos_test_case_x, pos_test_y) = UDPos.readDataset(windowSize, word2Idx, caseLookup)
    pos_n_out = pos_train_y_cat.shape[1]

    model_train_input_pos = [pos_train_x, pos_train_case_x]
    model_dev_input_pos = [pos_dev_x, pos_dev_case_x]
    model_test_input_pos = [pos_test_x, pos_test_case_x]

    ner_model = OptimizedModels.getNERModel(embeddings, word2Idx)
    #pred_train = pos_model.predict(model_train_input_pos, verbose=0).argmax(axis=-1)
    #pred_dev = pos_model.predict(model_dev_input_pos, verbose=0).argmax(axis=-1)
    pred_test = ner_model.predict(model_test_input_pos, verbose=0).argmax(axis=-1)

#extendCoNLLNer()

def buildAndTrainNERModel():
    [input_train, ner_train_y_cat], [input_dev, ner_dev_y], [input_test, ner_test_y], dicts = CoNLLNer.readDatasetExt(windowSize, word2Idx, caseLookup)

    [ner_train_x, ner_train_pos_x, ner_train_casing_x] = input_train
    [ner_dev_x, ner_dev_pos_x, ner_dev_casing_x] = input_dev
    [ner_test_x, ner_test_pos_x, ner_test_casing_x] = input_test
    [unused, pos2Idx, case2Idx] = dicts

    ner_n_out = ner_train_y_cat.shape[1]
    n_in_x = ner_train_x.shape[1]
    n_in_pos = ner_train_pos_x.shape[1]
    n_in_casing = ner_train_casing_x.shape[1]


    words_input = Input(shape=(n_in_x,), dtype='int32', name='words_input')
    wordEmbeddingLayer = Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in_x,
                                   weights=[embeddings], trainable=False)
    words = wordEmbeddingLayer(words_input)
    words = Flatten(name='words_flatten')(words)

    pos_input = Input(shape=(n_in_pos,), dtype='int32', name='pos_input')
    posEmbeddingLayer = Embedding(output_dim=len(pos2Idx), input_dim=len(pos2Idx), input_length=n_in_pos,
                                   trainable=True)
    pos = posEmbeddingLayer(pos_input)
    pos = Flatten(name='pos_flatten')(pos)

    case_input = Input(shape=(n_in_x,), dtype='int32', name='case_input')
    caseEmbeddingLayer = Embedding(output_dim=len(case2Idx), input_dim=len(case2Idx), input_length=n_in_casing,
                                   trainable=True)
    casing = caseEmbeddingLayer(case_input)
    casing = Flatten(name='casing_flatten')(casing)

    input_layers = [words, pos, casing]
    inputs = [words_input, pos_input, case_input]

    input_layers_merged = merge(input_layers, mode='concat')

    model_ner = NER.buildNERModelGivenInput(input_layers_merged, inputs, numHiddenUnitsNER, ner_n_out)

    dev_accs_ner, test_accs_ner, dev_f1s_ner, test_f1s_ner, ranges = Trainer.trainModelWithIncreasingData(model_ner, input_train, ner_train_y_cat, number_of_epochs, n_minibatches, input_dev, ner_dev_y, input_test, ner_test_y)

    return dev_accs_ner, test_accs_ner, dev_f1s_ner, test_f1s_ner, ranges

'''dev_accs_ner, test_accs_ner, dev_f1s_ner, test_f1s_ner = buildAndTrainNERModel()

metric_results.append((dev_accs_ner, 'ner_dev_acc'))
metric_results.append((test_accs_ner, 'ner_test_acc'))
metric_results.append((dev_f1s_ner, 'ner_dev_f1'))
metric_results.append((test_f1s_ner, 'ner_test_f1'))

LearningCurve.plotLearningCurve(metric_results)'''