import numpy as np
import pandas as pd
import os

DATA_EXP_PATH = os.environ.get('DATA_EXP_PATH', '../../data/data_experiments.csv')
DATA_SCENARIOS_PATH = os.environ.get('DATA_SCENARIO_PATH', '../../data/data_scenarios.csv')
RESULTS_FOLDER = os.environ.get('RESULTS_FOLDER', '../../results/')

pd.set_option('display.width', 180)

df_exp = pd.read_csv(DATA_EXP_PATH, sep=',', decimal='.')
df_exp.transfer = df_exp['transfer'].str.strip()
df_exp.exp = df_exp['exp'].str.strip()
df_exp.task = df_exp['task'].str.strip()

df_scenario = pd.read_csv(DATA_SCENARIOS_PATH, sep=',', decimal='.')
df_scenario.transfer = df_scenario['transfer'].str.strip()
df_scenario.exp = df_scenario['exp'].str.strip()
df_scenario.task = df_scenario['task'].str.strip()


def replaceTaskInTransferForMultitask(df, task, transfer_task = None):
    tr_task = transfer_task if transfer_task else task
    df.loc[(df.exp == 'multitask') & (df.task == task), 'transfer'] = \
    df[(df.exp == 'multitask') & (df.task == task)]['transfer'].str.replace(tr_task + '-', '')
    df.loc[(df.exp == 'multitask') & (df.task == task), 'transfer'] = \
        df[(df.exp == 'multitask') & (df.task == task)]['transfer'].str.replace('-' + tr_task, '')

# replace task name which also appears in the transfer column in the multitask exp
replaceTaskInTransferForMultitask(df_exp, 'ace')
replaceTaskInTransferForMultitask(df_exp, 'ecb')
replaceTaskInTransferForMultitask(df_exp, 'tac')
replaceTaskInTransferForMultitask(df_exp, 'tempeval')
replaceTaskInTransferForMultitask(df_exp, 'ner')
replaceTaskInTransferForMultitask(df_exp, 'chunking')
replaceTaskInTransferForMultitask(df_exp, 'wsj_pos', 'pos')

# delete source transfer combinations which are not used in the thesis
df_exp = df_exp[-df_exp.transfer.isin(['ecb-tac', 'ecb-tac-tempeval', 'tac-tempeval', 'ace-ecb', 'ace-ecb-tempeval', 'ace-tempeval'])]

def medianAndStdFunc(data):
    return u'{0:.2f}\u00B1{1:.2f}%'.format(np.median(data) * 100, np.std(data) * 100)

def avgAndStdFunc(data):
    return u'{0:.2f}\u00B1{1:.2f}%'.format(np.average(data) * 100, np.std(data) * 100)

current_metric = avgAndStdFunc

def get_results_for_task(df, task, source_tasks, aggfuncs, minSize, maxSize, pnn_arch=['pnn'], suffix=''):
    multitask_transfer_spec = map(lambda t: '-'.join(sorted(t.split('-') + [task])),source_tasks)
    tmp = df[(df.score_dev != 0) & (df.task == task) & (df.transfer.isin(source_tasks + ['none'] + multitask_transfer_spec)) & (df['sample'] >= minSize) & (df['sample'] <= maxSize) & (df['exp'].isin(['baseline','finetuning','pipeline','multitask'] + pnn_arch))]
    print 'Target task: {0}, source tasks: {1}, metric: {2}\n\n'.format(task, source_tasks, ','.join(map(lambda func: func.__name__,aggfuncs)))
    res = pd.pivot_table(tmp, index=['exp', 'transfer'], columns='sample', values='score_test', aggfunc=aggfuncs)
    print res
    print '\n---------------------------------------------------\n'
    res.to_csv(RESULTS_FOLDER + '/' + task + suffix + '.csv', encoding='utf-8')

def compare_methods(df, sample, aggfuncs):
    tmp = df[(df.score_dev != 0) & (df['sample'] == sample) & (-df['exp'].isin(['baseline','finetuning','pipeline','multitask']))]
    res = pd.pivot_table(tmp, index='exp', columns=['task', 'transfer'], values='score_test', aggfunc=aggfuncs)
    print res
    return res
    print tmp.groupby(['task', 'transfer'])['score_test'].agg(aggfuncs)

get_results_for_task(df_exp, 'tac', ['tempeval', 'ecb', 'ecb-tempeval'], [current_metric], 2000, 30000, ['pnn_adapter_10'], '_uncat')
get_results_for_task(df_exp, 'tac', ['ace'], [current_metric], 2000, 30000, ['pnn_adapter_100'], '_cat')
get_results_for_task(df_exp, 'ace', ['tempeval', 'ecb', 'ecb-tempeval'], [current_metric], 5000, 30000, ['pnn_adapter_10'], '_uncat')
get_results_for_task(df_exp, 'ace', ['tac'], [current_metric], 5000, 30000, ['pnn_adapter_100'], '_cat')
get_results_for_task(df_exp, 'ecb', ['tempeval'], [current_metric], 1000, 30000, ['pnn'])
get_results_for_task(df_exp, 'tempeval', ['ecb'], [current_metric], 1000, 30000, ['pnn'])

get_results_for_task(df_exp, 'wsj_pos', ['chunking', 'chunking-ner', 'ner', 'chunking-pos', 'chunking-ner-pos', 'ner-pos'], [current_metric], 2000, 30000, ['pnn_adapter_100'])
get_results_for_task(df_exp, 'wsj_pos', ['ud_pos'], [current_metric], 2000, 30000, ['pnn_adapter_50'], '_ud')
get_results_for_task(df_exp, 'ud_pos', ['pos', 'pos-ud_pos'], [current_metric], 2000, 30000, ['pnn_adapter_100'])
get_results_for_task(df_exp, 'ner', ['pos', 'chunking', 'chunking-pos'], [current_metric], 2000, 30000, ['pnn_simple_adapter_10'])
get_results_for_task(df_exp, 'chunking', ['pos', 'ner', 'ner-pos'], [current_metric], 2000, 30000, ['pnn_adapter_100'])

get_results_for_task(df_scenario, 'ace_only_contacts', ['ace_wo_contacts', 'ecb', 'tempeval'], [current_metric], 1000, 30000, ['pnn_adapter_10', 'pnn_adapter_50', 'pnn_adapter_100', 'pnn', 'pnn_simple_adapter_10', 'pnn_simple_adapter_50', 'pnn_simple_adapter_100', 'pnn_simple'])
get_results_for_task(df_scenario, 'ace_only_business', ['ace_wo_business', 'ecb', 'tempeval'], [current_metric], 1000, 30000, ['pnn_adapter_10', 'pnn_adapter_50', 'pnn_adapter_100', 'pnn', 'pnn_simple_adapter_10', 'pnn_simple_adapter_50', 'pnn_simple_adapter_100', 'pnn_simple'])
get_results_for_task(df_scenario, 'ace_only_justice', ['ace_wo_justice', 'ecb', 'tempeval'], [current_metric], 1000, 30000, ['pnn_adapter_10', 'pnn_adapter_50', 'pnn_adapter_100', 'pnn', 'pnn_simple_adapter_10', 'pnn_simple_adapter_50', 'pnn_simple_adapter_100', 'pnn_simple'])
get_results_for_task(df_scenario, 'ace_only_movement', ['ace_wo_movement', 'ecb', 'tempeval'], [current_metric], 1000, 30000, ['pnn_adapter_10', 'pnn_adapter_50', 'pnn_adapter_100', 'pnn', 'pnn_simple_adapter_10', 'pnn_simple_adapter_50', 'pnn_simple_adapter_100', 'pnn_simple'])

get_results_for_task(df_scenario, 'tac_newswire', ['tac_forum'], [current_metric], 5000, 30000, ['pnn_adapter_10', 'pnn_adapter_50', 'pnn_adapter_100', 'pnn'])
get_results_for_task(df_scenario, 'tac_forum', ['tac_newswire'], [current_metric], 5000, 100000, ['pnn_adapter_10', 'pnn_adapter_50', 'pnn_adapter_100', 'pnn'])

res = compare_methods(df_exp, 10000, current_metric)
res.to_csv(RESULTS_FOLDER + '/' + 'pnn_comp.csv', encoding='utf-8')



