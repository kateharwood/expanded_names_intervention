import numpy as np
from scipy.optimize import linprog
from scipy.linalg import block_diag


cost_matrix = np.load('data/euclidean_cost_matrix.npy')
female = 1500
male= 2500
cost_matrix = cost_matrix[:female,:male] # Change these based on how balanced/unbalanced we want
num_female_names, num_male_names = cost_matrix.shape

coefficients = cost_matrix.flatten()
ones = np.tile(np.ones((num_male_names)), (num_female_names, 1))
female_matches_constraints = block_diag(*ones)


identity = np.eye(num_male_names, num_male_names)
male_matches_constraints = np.tile(identity, (1, num_female_names))
lhs_eq = np.vstack((female_matches_constraints, male_matches_constraints))

# make sure the equality constraints have same number of variables as there are coefficients
assert(lhs_eq.shape[1] == coefficients.shape[0])
female_sum =  np.full((num_female_names, 1), male/female) # total probabilities for all female to male matches have to sum to ratio of male to female names
male_sum = np.ones((num_male_names, 1)) # total probabilities for all male to female matches have to sum to 1
rhs_eq = np.vstack((female_sum, male_sum))
bounds = (0, float("inf")) # probabilities have to be greater than 0

print('Done creating the constraints.')

# objective it is solving for is min sum over all coefficients in the cost matrix * target variables
opt = linprog(c=coefficients, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bounds, method='highs')
print(opt.x)
print(np.nonzero(opt.x))
print("Finished solving the LP.")
np.save('data/1500_2500_names_new.npy', opt.x)