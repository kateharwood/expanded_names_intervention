import os
import json
import jsonlines
import math
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

with open('data/name_frequencies.jsonl', 'r') as f:
    lines = f.read().splitlines()

female_names = []
male_names = []
for line in lines:
    line = json.loads(line)
    female_names.append(line)
    male_names.append(line)
# print(female_names[:600])

female_names.sort(key=lambda x: x[list(x.keys())[0]]['F'], reverse=True)
male_names.sort(key=lambda x: x[list(x.keys())[0]]['M'], reverse=True)


# make sure there are no overlaps in the two lists
top_female_names = []
top_male_names = []
for i,name in enumerate(female_names):
    if name[list(name.keys())[0]]['F'] > name[list(name.keys())[0]]['M']:
        top_female_names.append(name)
    # we don't need to include names that are equally male and female because those names do not need to be swapped
    elif name[list(name.keys())[0]]['F'] == name[list(name.keys())[0]]['M']: 
        print(name)
    else:
        top_male_names.append(name)

# replicating the number of names they used in the original paper
top_female_names = top_female_names[:2500]
top_male_names = top_male_names[:2500]

print("Done creating names lists.")

female_coords = np.zeros((len(top_female_names), 2), dtype=float)
male_coords = np.zeros((len(top_male_names), 2), dtype=float)
female_name_string = []
for i, name in enumerate(top_female_names):
    key = list(name.keys())[0]
    x = name[key]['F']
    y = name[key]['M']
    female_coords[i] = [x,y]
    female_name_string.append(key)
male_name_string = []
for i,name in enumerate(top_male_names):
    key = list(name.keys())[0]
    x = name[key]['M']
    y = name[key]['F']
    male_coords[i] = [x,y]
    male_name_string.append(key)

assert(len(male_coords) == len(female_coords) == 2500)
assert(set(female_name_string).isdisjoint(male_name_string))

with open('data/female_names.txt', 'w') as f:
    f.write('\n'.join(list(female_name_string)))
with open('data/male_names.txt', 'w') as f:
    f.write('\n'.join(list(male_name_string)))

cost_matrix = euclidean_distances(female_coords, male_coords)
np.save('data/euclidean_cost_matrix.npy', cost_matrix)