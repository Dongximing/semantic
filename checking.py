import pickle

with open('/data/ximing/math-result_left/data-500-temp0_0', "rb") as f:
    generations = pickle.load(f)
    for g in generations:
        if 'cluster_assignment_entropy' in g:
            print('cluster_assignment_entropy',g['cluster_assignment_entropy'])
        elif 'clustering-gpt-prompt' in g:
            print('clustering-gpt-prompt',g['clustering-gpt-prompt'])