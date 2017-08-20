import numpy as np
import pandas as pd

#RESULT_PATH = '/Users/lucas/Documents/Uni/Masterarbeit/Event_Nugget_Detection/Thesis/data/ed_results.csv'
ED_PATH = '/Users/lucas/Documents/Uni/Masterarbeit/Event_Nugget_Detection/Thesis/data/raw/results.csv'
STD_NLP_PATH = '/Users/lucas/Documents/Uni/Masterarbeit/Event_Nugget_Detection/Thesis/data/raw/results_standard.csv'
#RESULT_PATH = '/Users/lucas/Documents/Uni/Masterarbeit/Event_Nugget_Detection/experiments_nlp.csv'

df_ed = pd.read_csv(ED_PATH, sep=',', decimal='.')

df_ed.transfer = df_ed['transfer'].str.strip()
df_ed.exp = df_ed['exp'].str.strip()
df_ed.task = df_ed['task'].str.strip()

df_nlp = pd.read_csv(STD_NLP_PATH, sep=',', decimal='.')

df_nlp.transfer = df_nlp['transfer'].str.strip()
df_nlp.exp = df_nlp['exp'].str.strip()
df_nlp.task = df_nlp['task'].str.strip()
#df.set = df['set'].str.strip()

#df_test = df[df['sample'] <= 30000]
#df_test = df[df['sample'] <= 30000]

def medianAndStdFunc(data):
    return u'{0:.3f}\u00B1{1:.3f}%'.format(np.median(data) * 100, np.std(data) * 100)

def get_results_for_task(df, task, source_tasks, aggfuncs, minSize, maxSize):
    multitask_transfer_spec = map(lambda t: '-'.join(sorted(t.split('-') + [task])),source_tasks)
    tmp = df[(df.task == task) & (df.transfer.isin(source_tasks + ['none'] + multitask_transfer_spec)) & (df['sample'] >= minSize) & (df['sample'] <= maxSize)]
    print 'Target task: {0}, source tasks: {1}, metric: {2}\n\n'.format(task, source_tasks, ','.join(map(lambda func: func.__name__,aggfuncs)))
    print pd.pivot_table(tmp, index=['exp'], columns='sample', values='score_test', aggfunc=aggfuncs)
    #print pd.pivot_table(tmp, index=['exp'], columns='sample', values='score_test', aggfunc=aggfunc)
    print '\n---------------------------------------------------\n'

get_results_for_task(df_ed, 'tac', ['tempeval', 'ecb', 'ecb-tempeval'], [len], 2000, 30000)
get_results_for_task(df_ed, 'tac', ['ace'], [len], 2000, 30000)
get_results_for_task(df_ed, 'ace', ['tempeval', 'ecb', 'ecb-tempeval'], [len], 5000, 30000)
get_results_for_task(df_ed, 'ace', ['tac'], [len], 5000, 30000)
get_results_for_task(df_ed, 'ecb', ['tempeval'], [len], 1000, 30000)
get_results_for_task(df_ed, 'tempeval', ['ecb'], [len], 1000, 30000)

get_results_for_task(df_nlp, 'wsj_pos', ['ner', 'chunking', 'chunking-ner'], [len], 2000, 30000)
get_results_for_task(df_nlp, 'ner', ['pos', 'chunking', 'chunking-pos'], [len], 2000, 30000)
get_results_for_task(df_nlp, 'chunking', ['pos', 'ner', 'ner-pos'], [len], 2000, 30000)
#get_results_for_task(df_test, 'ace', ['tempeval', 'ecb'], [np.std])
#get_results_for_task(df_test, 'ace', ['tac'], [np.std])
#get_results_for_task(df_test, 'wsj_pos', ['ner', 'chunking'], np.median)
#get_results_for_task(df_test, 'tac', ['ace'], np.max)
#get_results_for_task(df_test, 'ace', ['tac'], np.median)



