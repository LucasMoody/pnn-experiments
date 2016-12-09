from embeddings.dependency_based_word_embeddings import DependencyBasedWordEmbeddings as Embeddings
from datasets.conll_ner import CoNLLNer
from models.NER import SennaNER as NER
from models import Trainer
from plots import LearningCurve

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
    # ----- NER ----- #

    (ner_train_x, ner_train_case_x, ner_train_y, ner_train_y_cat), (ner_dev_x, ner_dev_case_x, ner_dev_y), (
        ner_test_x, ner_test_case_x, ner_test_y) = CoNLLNer.readDataset(windowSize, word2Idx, caseLookup)
    ner_n_out = ner_train_y_cat.shape[1]

    model_train_input_ner = [ner_train_x, ner_train_case_x]
    model_dev_input_ner = [ner_dev_x, ner_dev_case_x]
    model_test_input_ner = [ner_test_x, ner_test_case_x]

    # ----- Build Model ----- #
    model_ner = NER.buildNERModel(n_in, embeddings, n_in_case, numHiddenUnitsNER, ner_n_out)

    print ner_train_x.shape[0], ' train samples'
    print ner_train_x.shape[1], ' train dimension'
    print ner_test_x.shape[0], ' test samples'


    # ----- Train Model ----- #
    dev_accs_ner, test_accs_ner, dev_f1s_ner, test_f1s_ner, ranges = Trainer.trainModelWithIncreasingData(model_ner, model_train_input_ner,
                                                                           ner_train_y_cat, number_of_epochs,
                                                                           n_minibatches, model_dev_input_ner,
                                                                           ner_dev_y, model_test_input_ner, ner_test_y)


    return dev_accs_ner, test_accs_ner, dev_f1s_ner, test_f1s_ner, ranges

'''dev_accs_ner, test_accs_ner, dev_f1s_ner, test_f1s_ner = buildAndTrainNERModel()

metric_results.append((dev_accs_ner, 'ner_dev_acc'))
metric_results.append((test_accs_ner, 'ner_test_acc'))
metric_results.append((dev_f1s_ner, 'ner_dev_f1'))
metric_results.append((test_f1s_ner, 'ner_test_f1'))

LearningCurve.plotLearningCurve(metric_results)'''