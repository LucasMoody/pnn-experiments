import numpy as np
import pandas as pd

pd.set_option('display.width', 180)

#RESULT_PATH = '/Users/lucas/Documents/Uni/Masterarbeit/Event_Nugget_Detection/Thesis/data/ed_results.csv'
ED_PATH = '/Users/lucas/Documents/Uni/Masterarbeit/Event_Nugget_Detection/Thesis/data/raw/results.csv'
STD_NLP_PATH = '/Users/lucas/Documents/Uni/Masterarbeit/Event_Nugget_Detection/Thesis/data/raw/results_standard.csv'
ALL_PATH = '/Users/lucas/Documents/Uni/Masterarbeit/Event_Nugget_Detection/Thesis/data/raw/all.csv'
SCENARIO_PATH = '/Users/lucas/Documents/Uni/Masterarbeit/Event_Nugget_Detection/Thesis/data/scenario/scenario.csv'
#RESULT_PATH = '/Users/lucas/Documents/Uni/Masterarbeit/Event_Nugget_Detection/experiments_nlp.csv'

df_all = pd.read_csv(ALL_PATH, sep=',', decimal='.')
df_all.transfer = df_all['transfer'].str.strip()
df_all.exp = df_all['exp'].str.strip()
df_all.task = df_all['task'].str.strip()

df_scenario = pd.read_csv(SCENARIO_PATH, sep=',', decimal='.')
df_scenario.transfer = df_scenario['transfer'].str.strip()
df_scenario.exp = df_scenario['exp'].str.strip()
df_scenario.task = df_scenario['task'].str.strip()


def replaceTaskInTransferForMultitask(df, task, transfer_task = None):
    tr_task = transfer_task if transfer_task else task
    df.loc[(df.exp == 'multitask') & (df.task == task), 'transfer'] = \
    df[(df.exp == 'multitask') & (df.task == task)]['transfer'].str.replace(tr_task + '-', '')
    df.loc[(df.exp == 'multitask') & (df.task == task), 'transfer'] = \
        df[(df.exp == 'multitask') & (df.task == task)]['transfer'].str.replace('-' + tr_task, '')

df_ed = pd.read_csv(ED_PATH, sep=',', decimal='.')

df_ed.transfer = df_ed['transfer'].str.strip()
df_ed.exp = df_ed['exp'].str.strip()
df_ed.task = df_ed['task'].str.strip()

df_nlp = pd.read_csv(STD_NLP_PATH, sep=',', decimal='.')

df_nlp.transfer = df_nlp['transfer'].str.strip()
df_nlp.exp = df_nlp['exp'].str.strip()
df_nlp.task = df_nlp['task'].str.strip()
#df.set = df['set'].str.strip()

# replace task name which also appears in the transfer column in the multitask exp
replaceTaskInTransferForMultitask(df_all, 'ace')
replaceTaskInTransferForMultitask(df_all, 'ecb')
replaceTaskInTransferForMultitask(df_all, 'tac')
replaceTaskInTransferForMultitask(df_all, 'tempeval')
replaceTaskInTransferForMultitask(df_all, 'ner')
replaceTaskInTransferForMultitask(df_all, 'chunking')
replaceTaskInTransferForMultitask(df_all, 'wsj_pos', 'pos')

# delete source transfer combinations which are not used in the thesis
df_all = df_all[-df_all.transfer.isin(['ecb-tac', 'ecb-tac-tempeval','tac-tempeval','ace-ecb','ace-ecb-tempeval','ace-tempeval'])]

#df_test = df[df['sample'] <= 30000]
#df_test = df[df['sample'] <= 30000]


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
    #print pd.pivot_table(tmp, index=['exp'], columns='sample', values='score_test', aggfunc=aggfunc)
    print res
    print '\n---------------------------------------------------\n'
    res.to_csv('/Users/lucas/Documents/Uni/Masterarbeit/Event_Nugget_Detection/Thesis/results/' + task + suffix + '.csv', encoding='utf-8')

def compare_methods(df, sample, aggfuncs):
    tmp = df[(df.score_dev != 0) & (df['sample'] == sample) & (-df['exp'].isin(['baseline','finetuning','pipeline','multitask']))]
    #print pd.pivot_table(tmp, index=['task', 'transfer'], columns='exp', values='score_test', aggfunc=aggfuncs)
    res = pd.pivot_table(tmp, index='exp', columns=['task', 'transfer'], values='score_test', aggfunc=aggfuncs)
    print res
    return res
    print tmp.groupby(['task', 'transfer'])['score_test'].agg(aggfuncs)

get_results_for_task(df_all, 'tac', ['tempeval', 'ecb', 'ecb-tempeval'], [current_metric], 2000, 30000, ['pnn_adapter_10'],'_uncat')
get_results_for_task(df_all, 'tac', ['ace'], [current_metric], 2000, 30000, ['pnn_adapter_100'],'_cat')
get_results_for_task(df_all, 'ace', ['tempeval', 'ecb', 'ecb-tempeval'], [current_metric], 5000, 30000, ['pnn_adapter_10'],'_uncat')
get_results_for_task(df_all, 'ace', ['tac'], [current_metric], 5000, 30000, ['pnn_adapter_100'],'_cat')
get_results_for_task(df_all, 'ecb', ['tempeval'], [current_metric], 1000, 30000, ['pnn'])
get_results_for_task(df_all, 'tempeval', ['ecb'], [current_metric], 1000, 30000, ['pnn'])

get_results_for_task(df_all, 'wsj_pos', ['chunking', 'chunking-ner', 'ner', 'chunking-pos', 'chunking-ner-pos', 'ner-pos'], [current_metric], 2000, 30000, ['pnn_adapter_100'])
get_results_for_task(df_all, 'wsj_pos', ['ud_pos'], [current_metric], 2000, 30000, ['pnn_adapter_50'],'_ud')
get_results_for_task(df_all, 'ud_pos', ['pos','pos-ud_pos'], [current_metric], 2000, 30000, ['pnn_adapter_100'])
get_results_for_task(df_all, 'ner', ['pos', 'chunking', 'chunking-pos'], [current_metric], 2000, 30000, ['pnn_simple_adapter_10'])
get_results_for_task(df_all, 'chunking', ['pos', 'ner', 'ner-pos'], [current_metric], 2000, 30000, ['pnn_adapter_100'])
#get_results_for_task(df_test, 'ace', ['tempeval', 'ecb'], [np.std])
#get_results_for_task(df_test, 'ace', ['tac'], [np.std])
#get_results_for_task(df_test, 'wsj_pos', ['ner', 'chunking'], np.median)
#get_results_for_task(df_test, 'tac', ['ace'], np.max)
#get_results_for_task(df_test, 'ace', ['tac'], np.median)

get_results_for_task(df_scenario, 'ace_only_contacts', ['ace_wo_contacts', 'ecb', 'tempeval'], [current_metric], 1000, 30000, ['pnn_adapter_10', 'pnn_adapter_50', 'pnn_adapter_100'])

res = compare_methods(df_all, 10000, current_metric)
res.to_csv('/Users/lucas/Documents/Uni/Masterarbeit/Event_Nugget_Detection/Thesis/results/pnn_comp.csv', encoding='utf-8')



