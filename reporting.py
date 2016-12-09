from experiments import exp1, exp_2, exp3, exp_4
from plots import LearningCurve

metric_results = []

dev_accs_ner_1, test_accs_ner_1, dev_f1s_ner_1, test_f1s_ner_1, ranges_1 = exp1.buildAndTrainNERModel()
dev_accs_ner_2, test_accs_ner_2, dev_f1s_ner_2, test_f1s_ner_2, ranges_2 = exp_2.buildAndTrainNERModel()
dev_accs_ner_3, test_accs_ner_3, dev_f1s_ner_3, test_f1s_ner_3, ranges_3 = exp3.buildAndTrainNERModel()
dev_accs_ner_4, test_accs_ner_4, dev_f1s_ner_4, test_f1s_ner_4, ranges_4 = exp_4.buildAndTrainNERModel()

metric_results.append((dev_f1s_ner_1, ranges_1, 'ner_dev_f1_1'))
metric_results.append((test_f1s_ner_1, ranges_1, 'ner_test_f1_1'))
metric_results.append((dev_f1s_ner_2, ranges_2, 'ner_dev_f1_2'))
metric_results.append((test_f1s_ner_2, ranges_2, 'ner_test_f1_2'))
metric_results.append((dev_f1s_ner_3, ranges_3, 'ner_dev_f1_3'))
metric_results.append((test_f1s_ner_3, ranges_3, 'ner_test_f1_3'))
metric_results.append((dev_f1s_ner_4, ranges_4, 'ner_dev_f1_4'))
metric_results.append((test_f1s_ner_4, ranges_4, 'ner_test_f1_4'))

LearningCurve.plotLearningCurve(metric_results)