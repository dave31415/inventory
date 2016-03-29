import numpy as np
import cvxpy as cvx
import scipy as scipy
import cvxopt as cvxopt

# Load time series data: S&P 500 price log.
y = np.loadtxt(open('data/snp500.txt', 'rb'), delimiter=",", skiprows=1)
n = y.size
# Form second difference matrix.
e = np.mat(np.ones((1, n)))
D = scipy.sparse.spdiags(np.vstack((e, -2*e, e)), range(3), n-2, n)
# Convert D to cvxopt sparse format, due to bug in scipy which prevents
# overloading neccessary for CVXPY. Use COOrdinate format as intermediate.
D_coo = D.tocoo()
D = cvxopt.spmatrix(D_coo.data, D_coo.row.tolist(), D_coo.col.tolist())

# Set regularization parameter.
vlambda = 50
# Solve l1 trend filtering problem.
x = cvx.Variable(n)
obj = cvx.Minimize(0.5 * cvx.sum_squares(y - x)
                   + vlambda * cvx.norm(D*x, 1) )
prob = cvx.Problem(obj)
# ECOS and SCS solvers fail to converge before
# the iteration limit. Use CVXOPT instead.
prob.solve(solver=cvx.CVXOPT,verbose=True)

print 'Solver status: ', prob.status
# Check for error.
if prob.status != cvx.OPTIMAL:
    raise Exception("Solver did not converge!")

import matplotlib.pyplot as plt

# Show plots inline in ipython.

# Plot properties.
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
plt.rc('font', **font)

# Plot estimated trend with original signal.
plt.figure(figsize=(6, 6))
plt.plot(np.arange(1,n+1), y, 'k:', linewidth=1.0)
plt.plot(np.arange(1,n+1), np.array(x.value), 'b-', linewidth=2.0)
plt.xlabel('date')
plt.ylabel('log price')

