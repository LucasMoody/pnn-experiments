from datasets.wsj_pos import WSJPos
from datasets.universal_dependencies_pos import UDPos
import datasets.conll_ner.CoNLLNer as CoNLLNer
import datasets.conll_chunking.CoNLLChunking as CoNLLChunking
from datasets.ace_ed import ACEED
from datasets.ace_ed import LabelFilter as ACELabelFilter
from datasets.tac2015_ed import TACED
from datasets.tempeval3_ed import TempevalED
from datasets.ecbplus_ed import ECBPlusED
from models import Trainer, InputBuilder, Senna
import numpy as np
import config

from embeddings.dependency_based_word_embeddings import DependencyBasedWordEmbeddings as Embeddings
from measurements import Measurer

# settings
wsj_pos_model_path = 'optimizer/saved_models/best_wsj_pos_96.09.hd5'
#wsj_pos_model_path = 'optimizer/saved_models/wsj_pos_no_dropout_95.96.hd5'
#wsj_pos_model_path = 'optimizer/saved_models/wsj_pos_300_95.93.hd5'
ud_pos_model_path = 'optimizer/saved_models/best_ud_pos_94.29.hd5'
ner_model_path = 'optimizer/saved_models/best_ner_87.94.hd5'
chunking_model_path = 'optimizer/saved_models/best_chunking_90.71.hd5'
ace_ed_model_path = 'optimizer/saved_models/ace_ed_64.12.hd5'
tac_ed_model_path = 'optimizer/saved_models/tac_57.75.hd5'
ecb_ed_model_path = 'optimizer/saved_models/ecbplus_ed_78.20.hd5'
tempeval_ed_model_path = 'optimizer/saved_models/tempeval_ed_82.65.hd5'
ace_wo_contacts_model_path = 'optimizer/saved_models/ace_without_contacts_62.58-49.62.hd5'
ace_wo_justice_model_path = 'optimizer/saved_models/ace_without_justice_62.34-55.20.hd5'
ace_wo_movement_model_path = 'optimizer/saved_models/ace_without_movement_60.77-47.27.hd5'
ace_wo_business_model_path = 'optimizer/saved_models/ace_without_business_61.69-57.32.hd5'
tac_newswire_ed_model_path = 'optimizer/saved_models/tac_newswire_49.80-45.12.hd5'
tac_forum_ed_model_path = 'optimizer/saved_models/tac_forum_45.33-45.47.hd5'

fixed_params_pos = {
    'update_word_embeddings': False,
    'window_size': 3,
    'batch_size': 128,
    'hidden_dims': 100,
    'activation': 'tanh',
    'dropout': 0.3,
    'optimizer': 'adam',
    'number_of_epochs': 12
}

fixed_params_chunking = {
    'update_word_embeddings': False,
    'window_size': 3,
    'batch_size': 128,
    'hidden_dims': 100,
    'activation': 'tanh',
    'dropout': 0.3,
    'optimizer': 'adam',
    'number_of_epochs': 5
}

fixed_params_ner = {
    'update_word_embeddings': False,
    'window_size': 3,
    'batch_size': 128,
    'hidden_dims': 100,
    'activation': 'tanh',
    'dropout': 0.3,
    'optimizer': 'adam',
    'number_of_epochs': 7
}

params_quick = {
    'update_word_embeddings': True,
    'window_size': 0,
    'batch_size': 128,
    'hidden_dims': 180,
    'activation': 'relu',
    'dropout': 0.25,
    'optimizer': 'nadam',
    'number_of_epochs': 1
}

params_pos_ws_0 = {
    'update_word_embeddings': True,
    'window_size': 0,
    'batch_size': 128,
    'hidden_dims': 180,
    'activation': 'relu',
    'dropout': 0.25,
    'optimizer': 'nadam',
    'number_of_epochs': 11
}

params_pos_ws_1 = {
    'update_word_embeddings': True,
    'window_size': 1,
    'batch_size': 128,
    'hidden_dims': 300,
    'activation': 'relu',
    'dropout': 0.25,
    'optimizer': 'adamax',
    'number_of_epochs': 13
}

params_pos_ws_2 = {
    'update_word_embeddings': True,
    'window_size': 2,
    'batch_size': 32,
    'hidden_dims': 230,
    'activation': 'sigmoid',
    'dropout': 0.45,
    'optimizer': 'nadam',
    'number_of_epochs': 11
}

params_pos_ws_3 = {
    'update_word_embeddings': False,
    'window_size': 3,
    'batch_size': 64,
    'hidden_dims': 280,
    'activation': 'relu',
    'dropout': 0.6,
    'optimizer': 'adamax',
    'number_of_epochs': 19
}

params_pos_ws_4 = {
    'update_word_embeddings': False,
    'window_size': 4,
    'batch_size': 128,
    'hidden_dims': 255,
    'activation': 'sigmoid',
    'dropout': 0.65,
    'optimizer': 'nadam',
    'number_of_epochs': 19
}

params_ner_ws_0 = {
    'update_word_embeddings': True,
    'window_size': 0,
    'batch_size': 64,
    'hidden_dims': 185,
    'activation': 'relu',
    'dropout': 0.1,
    'optimizer': 'adam',
    'number_of_epochs': 14
}

params_ner_ws_1 = {
    'update_word_embeddings': False,
    'window_size': 1,
    'batch_size': 128,
    'hidden_dims': 235,
    'activation': 'relu',
    'dropout': 0.5,
    'optimizer': 'adam',
    'number_of_epochs': 16
}

params_ner_ws_2 = {
    'update_word_embeddings': True,
    'window_size': 2,
    'batch_size': 32,
    'hidden_dims': 270,
    'activation': 'sigmoid',
    'dropout': 0.4,
    'optimizer': 'adam',
    'number_of_epochs': 19
}

params_ner_ws_3 = {
    'update_word_embeddings': True,
    'window_size': 3,
    'batch_size': 32,
    'hidden_dims': 175,
    'activation': 'sigmoid',
    'dropout': 0.45,
    'optimizer': 'adam',
    'number_of_epochs': 19
}

params_ner_ws_4 = {
    'update_word_embeddings': False,
    'window_size': 4,
    'batch_size': 32,
    'hidden_dims': 190,
    'activation': 'sigmoid',
    'dropout': 0.5,
    'optimizer': 'nadam',
    'number_of_epochs': 9
}

params_ed_ws_0 = {
    'update_word_embeddings': False,
    'window_size': 4,
    'batch_size': 32,
    'hidden_dims': 190,
    'activation': 'sigmoid',
    'dropout': 0.5,
    'optimizer': 'nadam',
    'number_of_epochs': 9
}

pos_default_params = {
    0: params_pos_ws_0,
    1: params_pos_ws_1,
    2: params_pos_ws_2,
    3: params_pos_ws_3,
    4: params_pos_ws_4
}

ner_default_params = {
    0: params_ner_ws_0,
    1: params_ner_ws_1,
    2: params_ner_ws_2,
    3: params_ner_ws_3,
    4: params_ner_ws_4
}

metrics = []

# ----- metric results -----#
metric_results = []

#Casing matrix
case2Idx = {
    'numeric': 0,
    'allLower': 1,
    'allUpper': 2,
    'initialUpper': 3,
    'other': 4,
    'mainly_numeric': 5,
    'contains_digit': 6,
    'PADDING': 7
}
n_in_case = len(case2Idx)

word2Idx = Embeddings.word2Idx
embeddings = Embeddings.embeddings


def getNERModel(learning_params=None):
    return getModelHelper(CoNLLNer.readDataset,
                             Measurer.create_compute_BIOf1,
                             name_prefix='ner_',
                             learning_params=learning_params)


def getWSJPOSModel(learning_params=None):
    return getModelHelper(WSJPos.readDataset,
                          lambda _: Measurer.measureAccuracy,
                          name_prefix='wsj_pos_',
                          learning_params=learning_params)


def getUDPOSModel(learning_params=None):
    return getModelHelper(UDPos.readDataset,
                          lambda _: Measurer.measureAccuracy,
                          name_prefix='ud_pos_',
                          learning_params=learning_params)


def getChunkingModel(learning_params=None):
    return getModelHelper(CoNLLChunking.readDataset,
                          Measurer.create_compute_BIOf1,
                          name_prefix='chunking_',
                          learning_params=learning_params)


def getAceEDModel(learning_params=None):
    return getModelHelper(ACEED.readDataset,
                          Measurer.create_compute_BIOf1,
                          name_prefix='ace_',
                          learning_params=learning_params)

def getAceWoContactsEDModel(learning_params=None):
    return getModelHelper(ACEED.readFilteredDataset,
                          Measurer.create_compute_BIOf1,
                          name_prefix='ace_without_contacts_',
                          learning_params=learning_params)

def getAceWoContactsEDModel(learning_params=None):
    def reader(window_size, word2Idx, case2Idx):
        return ACEED.readFilteredDataset(window_size, word2Idx, case2Idx, ACELabelFilter.wo_contacts_label_filter, dataset_filter_creator(ACELabelFilter.wo_contacts_label_filter))
    return getModelHelper(reader,
                          Measurer.create_compute_BIOf1,
                          name_prefix='ace_without_contacts_',
                          learning_params=learning_params)

def getAceWoBusinessEDModel(learning_params=None):
    def reader(window_size, word2Idx, case2Idx):
        return ACEED.readFilteredDataset(window_size, word2Idx, case2Idx, ACELabelFilter.wo_business_label_filter, dataset_filter_creator(ACELabelFilter.wo_business_label_filter))
    return getModelHelper(reader,
                          Measurer.create_compute_BIOf1,
                          name_prefix='ace_without_business_',
                          learning_params=learning_params)

def getAceWoJusticeEDModel(learning_params=None):
    def reader(window_size, word2Idx, case2Idx):
        return ACEED.readFilteredDataset(window_size, word2Idx, case2Idx, ACELabelFilter.wo_justice_label_filter, dataset_filter_creator(ACELabelFilter.wo_justice_label_filter))
    return getModelHelper(reader,
                          Measurer.create_compute_BIOf1,
                          name_prefix='ace_without_justice_',
                          learning_params=learning_params)

def getAceWoMovementEDModel(learning_params=None):
    def reader(window_size, word2Idx, case2Idx):
        return ACEED.readFilteredDataset(window_size, word2Idx, case2Idx, ACELabelFilter.wo_movement_label_filter, dataset_filter_creator(ACELabelFilter.wo_movement_label_filter))
    return getModelHelper(reader,
                          Measurer.create_compute_BIOf1,
                          name_prefix='ace_without_movement_',
                          learning_params=learning_params)


def getTacEDModel(learning_params=None):
    return getModelHelper(TACED.readDataset,
                          Measurer.create_compute_BIOf1,
                          name_prefix='tac_',
                          learning_params=learning_params)

def reader_domain_creator(domain):
    def reader(window_size, word2Idx, case2Idx):
        return TACED.readDomainDataset(window_size, word2Idx, case2Idx, domain)
    return reader

def getTacNewswireEDModel(learning_params=None):
    return getModelHelper(reader_domain_creator('newswire'),
                          Measurer.create_compute_BIOf1,
                          name_prefix='tac_newswire_',
                          learning_params=learning_params)

def getTacForumEDModel(learning_params=None):
    return getModelHelper(reader_domain_creator('forum'),
                          Measurer.create_compute_BIOf1,
                          name_prefix='tac_forum_',
                          learning_params=learning_params)


def getTempevalEDModel(learning_params=None):
    return getModelHelper(TempevalED.readDataset,
                          Measurer.create_compute_BIOf1,
                          name_prefix='tempeval_',
                          learning_params=learning_params)


def getEcbPlusEDModel(learning_params=None):
    return getModelHelper(ECBPlusED.readDataset,
                          Measurer.create_compute_BIOf1,
                          name_prefix='ecb_',
                          learning_params=learning_params)


def getWSJPOSModelGivenInput(input_layers,
                             inputs,
                             learning_params=None,
                             use_existing_model=True):
    return getModelGivenInputHelper(
        WSJPos.readDataset,
        input_layers,
        inputs,
        lambda _: Measurer.measureAccuracy,
        name_prefix='ws_pos_',
        learning_params=learning_params,
        model_path=wsj_pos_model_path)


def getUDPOSModelGivenInput(input_layers,
                            inputs,
                            learning_params=None,
                            window_size=None,
                            use_existing_model=True):
    return getModelGivenInputHelper(
        UDPos.readDataset,
        input_layers,
        inputs,
        lambda _: Measurer.measureAccuracy,
        name_prefix='ud_pos_',
        learning_params=learning_params,
        model_path=ud_pos_model_path)


def getNERModelGivenInput(input_layers,
                          inputs,
                          learning_params=None,
                          window_size=None,
                          use_existing_model=True):
    return getModelGivenInputHelper(
        CoNLLNer.readDataset,
        input_layers,
        inputs,
        Measurer.create_compute_BIOf1,
        name_prefix='ner_',
        learning_params=learning_params,
        model_path=ner_model_path)


def getChunkingModelGivenInput(input_layers,
                               inputs,
                               learning_params=None,
                               window_size=None,
                               use_existing_model=True):
    return getModelGivenInputHelper(
        CoNLLChunking.readDataset,
        input_layers,
        inputs,
        Measurer.create_compute_BIOf1,
        name_prefix='chunking_',
        learning_params=learning_params,
        model_path=chunking_model_path)


def dataset_filter_creator(label_filter):
    def dataset_filter(dataset):
        def contains_labels(sentence):
            return reduce(lambda result, word: result and label_filter(word[1]), sentence, True)
        return filter(contains_labels, dataset)
    return dataset_filter

def getAceEDModelGivenInput(input_layers,
                            inputs,
                            learning_params=None,
                            window_size=None,
                            use_existing_model=True):
    return getModelGivenInputHelper(
        ACEED.readDataset,
        input_layers,
        inputs,
        Measurer.create_compute_BIOf1,
        name_prefix='ace_ed_',
        learning_params=learning_params,
        model_path=ace_ed_model_path)

def getAceWoContactsModelGivenInput(input_layers,
                            inputs,
                            learning_params=None,
                            window_size=None,
                            use_existing_model=True):
    def reader(window_size, word2Idx, case2Idx):
        return ACEED.readFilteredDataset(window_size, word2Idx, case2Idx, ACELabelFilter.wo_contacts_label_filter, dataset_filter_creator(ACELabelFilter.wo_contacts_label_filter))
    return getModelGivenInputHelper(
        reader,
        input_layers,
        inputs,
        Measurer.create_compute_BIOf1,
        name_prefix='ace_wo_contacts_',
        learning_params=learning_params,
        model_path=ace_wo_contacts_model_path)

def getAceWoMovementModelGivenInput(input_layers,
                            inputs,
                            learning_params=None,
                            window_size=None,
                            use_existing_model=True):
    def reader(window_size, word2Idx, case2Idx):
        return ACEED.readFilteredDataset(window_size, word2Idx, case2Idx, ACELabelFilter.wo_movement_label_filter, dataset_filter_creator(ACELabelFilter.wo_movement_label_filter))
    return getModelGivenInputHelper(
        reader,
        input_layers,
        inputs,
        Measurer.create_compute_BIOf1,
        name_prefix='ace_wo_movement_',
        learning_params=learning_params,
        model_path=ace_wo_movement_model_path)

def getAceWoBusinessModelGivenInput(input_layers,
                            inputs,
                            learning_params=None,
                            window_size=None,
                            use_existing_model=True):
    def reader(window_size, word2Idx, case2Idx):
        return ACEED.readFilteredDataset(window_size, word2Idx, case2Idx, ACELabelFilter.wo_business_label_filter, dataset_filter_creator(ACELabelFilter.wo_business_label_filter))
    return getModelGivenInputHelper(
        reader,
        input_layers,
        inputs,
        Measurer.create_compute_BIOf1,
        name_prefix='ace_wo_business_',
        learning_params=learning_params,
        model_path=ace_wo_business_model_path)

def getAceWoJusticeModelGivenInput(input_layers,
                            inputs,
                            learning_params=None,
                            window_size=None,
                            use_existing_model=True):
    def reader(window_size, word2Idx, case2Idx):
        return ACEED.readFilteredDataset(window_size, word2Idx, case2Idx, ACELabelFilter.wo_justice_label_filter, dataset_filter_creator(ACELabelFilter.wo_justice_label_filter))
    return getModelGivenInputHelper(
        reader,
        input_layers,
        inputs,
        Measurer.create_compute_BIOf1,
        name_prefix='ace_wo_justice_',
        learning_params=learning_params,
        model_path=ace_wo_justice_model_path)

def getEcbEDModelGivenInput(input_layers,
                            inputs,
                            learning_params=None,
                            window_size=None,
                            use_existing_model=True):
    return getModelGivenInputHelper(
        ECBPlusED.readDataset,
        input_layers,
        inputs,
        Measurer.create_compute_BIOf1,
        name_prefix='ecb_ed_',
        learning_params=learning_params,
        model_path=ecb_ed_model_path)


def getTacEDModelGivenInput(input_layers,
                            inputs,
                            learning_params=None,
                            window_size=None,
                            use_existing_model=True):
    return getModelGivenInputHelper(
        TACED.readDataset,
        input_layers,
        inputs,
        Measurer.create_compute_BIOf1,
        name_prefix='tac_ed_',
        learning_params=learning_params,
        model_path=tac_ed_model_path)

def getTacNewswireEDModelGivenInput(input_layers,
                            inputs,
                            learning_params=None,
                            window_size=None,
                            use_existing_model=True):
    return getModelGivenInputHelper(
        reader_domain_creator('newswire'),
        input_layers,
        inputs,
        Measurer.create_compute_BIOf1,
        name_prefix='tac_newswire_ed_',
        learning_params=learning_params,
        model_path=tac_newswire_ed_model_path)

def getTacForumEDModelGivenInput(input_layers,
                            inputs,
                            learning_params=None,
                            window_size=None,
                            use_existing_model=True):
    return getModelGivenInputHelper(
        reader_domain_creator('forum'),
        input_layers,
        inputs,
        Measurer.create_compute_BIOf1,
        name_prefix='tac_forum_ed_',
        learning_params=learning_params,
        model_path=tac_forum_ed_model_path)

def getTempevalEDModelGivenInput(input_layers,
                                 inputs,
                                 learning_params=None,
                                 window_size=None,
                                 use_existing_model=True):
    return getModelGivenInputHelper(
        TempevalED.readDataset,
        input_layers,
        inputs,
        Measurer.create_compute_BIOf1,
        name_prefix='tempeval_ed_',
        learning_params=learning_params,
        model_path=tempeval_ed_model_path)


def getModelGivenInputHelper(reader,
                             input_layers,
                             inputs,
                             measurer_creator,
                             name_prefix='',
                             learning_params=None,
                             model_path=None):
    if learning_params is None:
        #params = ner_default_params[window_size]
        #params['number_of_epochs'] = 1
        params = fixed_params_chunking
        # todo throw error
    else:
        params = learning_params
    print params

    [input_train, train_y_cat], [input_dev, dev_y], [input_test, test_y], dicts = reader(params['window_size'], word2Idx, case2Idx)
    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, _, label2Idx, idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    # ----- Build Model ----- #
    model = Senna.buildModelGivenInput(
        input_layers, inputs, params, n_out, name_prefix=name_prefix)

    print train_x.shape[0], ' train samples'
    print train_x.shape[1], ' train dimension'
    print test_x.shape[0], ' test samples'

    # ----- Train Model ----- #
    measurer = measurer_creator(idx2Label)
    if model_path is not None:
        print 'Weight sum before setting weights:', reduce(
            lambda a, b: a + np.sum(b), model.get_weights(), 0)
        model.load_weights(model_path)
        print 'Weight sum after setting weights:', reduce(
            lambda a, b: a + np.sum(b), model.get_weights(), 0)
        pred_dev = model.predict(
            input_dev, verbose=0).argmax(axis=-1)  # Prediction of the classes
        print '{0} model has f1: {1:4f}'.format(
            name_prefix, measurer(pred_dev, dev_y) * 100)
    else:
        res = Trainer.trainModel(
            model,
            input_train,
            train_y_cat,
            params['number_of_epochs'],
            params['batch_size'],
            input_dev,
            dev_y,
            input_test,
            test_y,
            measurer=measurer)
    return model

def getModelHelper(reader,
                             measurer_creator,
                             name_prefix='',
                             learning_params=None):
    # load params
    if learning_params is None:
        params = params_pos_ws_0
    else:
        params = learning_params

    # load dataset
    [input_train,
     train_y_cat], [input_dev,
                    dev_y], [input_test, test_y], dicts = reader(
        params['window_size'], word2Idx, case2Idx)
    [train_x, train_case_x] = input_train
    [dev_x, dev_case_x] = input_dev
    [test_x, test_case_x] = input_test
    [_, caseLookup, label2Idx, idx2Label] = dicts
    n_out = train_y_cat.shape[1]

    n_in_x = train_x.shape[1]
    n_in_casing = train_case_x.shape[1]

    # ----- Build Model ----- #
    input_layers, inputs = InputBuilder.buildStandardModelInput(
        embeddings, case2Idx, n_in_x, n_in_casing)
    model = Senna.buildModelGivenInput(
        input_layers, inputs, params, n_out, name_prefix=name_prefix)

    print train_x.shape[0], ' train samples'
    print train_x.shape[1], ' train dimension'
    print test_x.shape[0], ' test samples'

    # ----- Train Model ----- #
    measurer = measurer_creator(idx2Label)
    best_train_score, best_dev_score, best_test_score, best_epoch = Trainer.trainModel(
        model,
        input_train,
        train_y_cat,
        params['number_of_epochs'],
        params['batch_size'],
        input_dev,
        dev_y,
        input_test,
        test_y,
        measurer=measurer)
    model.save_weights('optimizer/saved_models/{0}{1:.2f}-{2:.2f}.hd5'.format(
        name_prefix, best_dev_score * 100, best_test_score * 100))
    return best_train_score, best_dev_score, best_test_score, best_epoch
