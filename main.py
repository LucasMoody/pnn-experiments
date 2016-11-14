import datasets.universal_dependencies_pos.UDPos as UDPos
import datasets.conll_ner.CoNLLNer as CoNLLNer
import embeddings.dependency_based_word_embeddings.DependencyBasedWordEmbeddings as Embeddings
import numpy as np
import models.POS.SennaPOS as POS
import models.NER.SennaNER as NER
import models.Trainer as Trainer
from measurements import Measurer
import plots.LearningCurve as LearningCurve

# settings
windowSize = 3 # n to the left, n to the right
n_in = 2 * windowSize + 1
numHiddenUnitsPOS = 100
numHiddenUnitsNER = 10
n_minibatches = 2
number_of_epochs = 1

#Casing matrix
caseLookup = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING':7}
n_in_case = len(caseLookup)

# Read in embeddings
embeddings, word2Idx = Embeddings.readEmbeddings()

# ----- POS ----- #

# Read in files
(pos_train_x, pos_train_case_x, pos_train_y, pos_train_y_cat), (pos_dev_x, pos_dev_case_x, pos_dev_y), (
    pos_test_x, pos_test_case_x, pos_test_y) = UDPos.readDataset(windowSize, word2Idx, caseLookup)
pos_n_out = pos_train_y_cat.shape[1]

model_pos = POS.buildPosModel(n_in, embeddings, n_in_case, numHiddenUnitsPOS, pos_n_out)

print pos_train_x.shape[0], ' train samples'
print pos_train_x.shape[1], ' train dimension'
print pos_test_x.shape[0], ' test samples'

print "POS Hidden weights sum (before Train): ", np.sum(model_pos.get_layer(name='pos_hidden').W.get_value())


minibatch_size_pos = len(pos_train_x) / n_minibatches
#minibatch_size_pos = 128

dev_scores_pos = []
test_scores_pos = []

model_input_pos = [pos_train_x, pos_train_case_x]
on_batch_callback_pos = Measurer.createBatchCallback(model_pos, [pos_dev_x, pos_dev_case_x], pos_dev_y,
                                                [pos_test_x, pos_test_case_x], pos_test_y,
                                                dev_scores_pos, test_scores_pos)
Trainer.trainModel(model_pos, model_input_pos, pos_train_y_cat, number_of_epochs,
                                                     minibatch_size_pos, [pos_dev_x, pos_dev_case_x], pos_dev_y,
                                                     [pos_test_x, pos_test_case_x], pos_test_y,
                                                     callbacks=[on_batch_callback_pos])
print "POS Hidden weights sum (after POS train): ", np.sum(model_pos.get_layer(name='pos_hidden').W.get_value())


# ----- NER ----- #

(ner_train_x, ner_train_case_x, ner_train_y, ner_train_y_cat), (ner_dev_x, ner_dev_case_x, ner_dev_y), (
    ner_test_x, ner_test_case_x, ner_test_y) = CoNLLNer.readDataset(windowSize, word2Idx, caseLookup)
ner_n_out = ner_train_y_cat.shape[1]

model_ner = NER.buildNERModel(n_in, embeddings, n_in_case, numHiddenUnitsNER, ner_n_out)

print ner_train_x.shape[0], ' train samples'
print ner_train_x.shape[1], ' train dimension'
print ner_test_x.shape[0], ' test samples'

print "NER Hidden weights sum (before Train): ", np.sum(model_ner.get_layer(name='ner_hidden').W.get_value())

minibatch_size_ner = len(ner_train_x) / n_minibatches
#minibatch_size_ner = 128

dev_scores_ner = []
test_scores_ner = []

model_input_ner = [ner_train_x, ner_train_case_x]
on_batch_callback_ner = Measurer.createBatchCallback(model_ner, [ner_dev_x, ner_dev_case_x], ner_dev_y,
                                                [ner_test_x, ner_test_case_x], ner_test_y,
                                                dev_scores_ner, test_scores_ner)
Trainer.trainModel(model_ner, model_input_ner, ner_train_y_cat, number_of_epochs,
                                                     minibatch_size_ner, [ner_dev_x, ner_dev_case_x], ner_dev_y,
                                                     [ner_test_x, ner_test_case_x], ner_test_y,
                                                     callbacks=[on_batch_callback_ner])
print "NER Hidden weights sum (after NER train): ", np.sum(model_ner.get_layer(name='ner_hidden').W.get_value())

# ----- Plotting ----- #
dev_scores_pos.insert(0, 0)
test_scores_pos.insert(0, 0)

dev_scores_ner.insert(0, 0)
test_scores_ner.insert(0, 0)
LearningCurve.plotLearningCurve([(dev_scores_pos, 'pos_dev'), (test_scores_pos, 'pos_test'), (dev_scores_ner, 'ner_dev'), (test_scores_ner, 'ner_test')])
