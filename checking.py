import pickle

with open('/data/ximing/math-result_left/data-500-temp0_0/ww0.pkl', "rb") as f:
    generations = pickle.load(f)
    for g in generations:
        if 'cluster_assignment_entropy_deberta' in g:
            print('cluster_assignment_entropy',g['cluster_assignment_entropy_deberta'])
        elif 'clustering-gpt-prompty_deberta' in g:
            print('clustering-gpt-prompty_deberta',g['clustering-gpt-prompty_deberta'])