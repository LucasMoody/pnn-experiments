import numpy as np
import pandas as pd

#RESULT_PATH = '/Users/lucas/Documents/Uni/Masterarbeit/Event_Nugget_Detection/Thesis/data/ed_results.csv'
RESULT_PATH = '/Users/lucas/Documents/Uni/Masterarbeit/Event_Nugget_Detection/Temp/results_ed.csv'
#RESULT_PATH = '/Users/lucas/Documents/Uni/Masterarbeit/Event_Nugget_Detection/experiments_nlp.csv'

df = pd.read_csv(RESULT_PATH, sep=';', decimal=',')

df.transfer = df['transfer'].str.strip()
df.exp = df['exp'].str.strip()
df.task = df['task'].str.strip()
df.set = df['set'].str.strip()

df_test = df[(df['set'] == 'test') & (df['sample'] <= 30000)]
#df_test = df[df['sample'] <= 30000]

def get_results_for_task(df, task, source_tasks, aggfuncs):
    tmp = df[(df.task == task) & (df.transfer.isin(source_tasks + ['none']))]
    print 'Target task: {0}, source tasks: {1}, metric: {2}\n\n'.format(task, source_tasks, ','.join(map(lambda func: func.__name__,aggfuncs)))
    print pd.pivot_table(tmp, index=['exp'], columns='sample', values='score', aggfunc=aggfuncs)
    #print pd.pivot_table(tmp, index=['exp'], columns='sample', values='score_test', aggfunc=aggfunc)
    print '\n---------------------------------------------------\n'

get_results_for_task(df_test, 'tac', ['tempeval', 'ecb'], [np.std])
get_results_for_task(df_test, 'tac', ['ace'], [np.std])
#get_results_for_task(df_test, 'ace', ['tempeval', 'ecb'], [np.std])
#get_results_for_task(df_test, 'ace', ['tac'], [np.std])
#get_results_for_task(df_test, 'wsj_pos', ['ner', 'chunking'], np.median)
#get_results_for_task(df_test, 'tac', ['ace'], np.max)
#get_results_for_task(df_test, 'ace', ['tac'], np.median)



