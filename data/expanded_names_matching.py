import numpy as np
from scipy.optimize import linprog
from scipy.linalg import block_diag


cost_matrix = np.load('data/euclidean_cost_matrix.npy')
cost_matrix = cost_matrix[:1500,:2500] # Change these based on how balanced/unbalanced we want
cost_matrix_flipped = np.flip(cost_matrix)

num_female_names, num_male_names = cost_matrix.shape
coefficients = np.hstack((cost_matrix.flatten(), cost_matrix_flipped.flatten()))

female_matching_eq = []
female_name_index = 0
while female_name_index < num_female_names:
    female_matching_eq.append([])
    for i in range(num_male_names*(female_name_index)):
        female_matching_eq[female_name_index].append(0)
    for i in range(num_male_names):
        female_matching_eq[female_name_index].append(1)
    for i in range((num_male_names)*(num_female_names-(female_name_index+1))):
        female_matching_eq[female_name_index].append(0)
    female_name_index += 1
female_matching_eq = np.asarray(female_matching_eq)

male_matching_eq = []
male_name_index = 0
while male_name_index < num_male_names:
    male_matching_eq.append([])
    for i in range(num_female_names*(male_name_index)):
        male_matching_eq[male_name_index].append(0)
    for i in range(num_female_names):
        male_matching_eq[male_name_index].append(1)
    for i in range(num_female_names*(num_male_names-(male_name_index+1))):
        male_matching_eq[male_name_index].append(0)
    male_name_index += 1
male_matching_eq = np.asarray(male_matching_eq)

lhs_eq = block_diag(female_matching_eq, male_matching_eq)
print('Finished constructing equality constraint matrix.')

# make sure the equality constraints have same number of variables as there are coefficients
assert(lhs_eq.shape[1] == coefficients.shape[0])
rhs_eq = np.ones((num_female_names + num_male_names, 1)) # total probabilities have to sum to 1
bounds = (.00001, float("inf")) # probabilities have to be greater than 0

# objective it is solving for is min sum over all coefficients in the cost matrix * target variables
opt = linprog(c=coefficients, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bounds, method='highs') # TODO can try another method
print(opt.x)
print("Finished solving the LP.")
np.save('data/1500_2500_names.npy', opt.x)