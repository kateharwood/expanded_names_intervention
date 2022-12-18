import numpy as np
import scipy
import cupy as np


cost_matrix = np.load('data/euclidean_cost_matrix_2000.npy')
assignment_matrix = scipy.optimize.linear_sum_assignment(cost_matrix.get())
np.save('data/bipartite_assignment_matrix_2000.npy', assignment_matrix)

assignment_matrix = np.load('data/bipartite_assignment_matrix_2000.npy')

with open('data/female_names.txt', 'r') as f:
    female_names = f.readlines()
female_names = [name.strip('\n') for name in female_names]

with open('data/male_names.txt', 'r') as f:
    male_names = f.readlines()
male_names = [name.strip('\n') for name in male_names]

matches = []
for i, name in enumerate(female_names):
    match = assignment_matrix[1][i]
    name_match = male_names[match.item()]
    matches.append(name.lower() + " " + name_match.lower())
    if name_match == name:
        print(name)
    assert(name_match != name)

with open('data/bipartite_name_matches_2000.txt', 'w') as f:
    f.write('\n'.join(matches))





