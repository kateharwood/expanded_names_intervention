# TODO
# get euc cost matrix for uneven number of them
# those are the coefficients of the summation
# that summation is the minimization objective
# lhs_ineq is a matrix of shape (num female names + num male names, num female names + num male names), but some entries are 0 (for female/female and male/male matches)
# rhs_ineq is a vector of all ones of shape (num female names + num_male names, 1)
# opt = linprog(c=obj, A_ub=luh_ineq, b_ub=rhs_ineq, method=TODO)


# replicate results for equal number of female and male names using new method
def equal_number_names():
    cost_matrix = np.load('euclidean_cost_matrix.npy')
    




# def uneven_number_names():