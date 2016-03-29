from cvxpy import Variable, Minimize, Problem
import numpy as np
import scipy
import cvxopt
from matplotlib import pylab as plt

# one product, simplest supply chain


def identity_spmatrix(n):
    return cvxopt.spmatrix(1.0, range(n), range(n))


def spmatrix2np(spmat):
    """
        Convert a matrix or spmatrix to numpy 2D array
    :param spmat: matrix or spmatrix
    :return: numpy 2D array of type float64
    """
    return np.asarray(cvxopt.matrix(spmat)).squeeze()


def first_deriv_matrix(n):
    # a matrix which computes the first derivative of a vector
    # copied from examples on cvxpy page
    e = np.mat(np.ones((1, n)))
    D = scipy.sparse.spdiags(np.vstack((-e, e)), range(2), n-1, n)
    D_coo = D.tocoo()
    D = cvxopt.spmatrix(D_coo.data, D_coo.row.tolist(), D_coo.col.tolist())
    return D


def get_step_function_matrix(n):
    """
    Upper/lower triangular with all ones
    Can be used for cumulative sum with matrix op
    :param n:
    :return:
    """
    step = identity_spmatrix(n)
    for i in xrange(n):
        for j in xrange(n):
            if i >= j:
                step[i, j] = 1.0
    return step


def get_time_shift_matrix(n, shift):
    return cvxopt.spmatrix(1.0, shift+np.arange(n), np.arange(n))


def create_schedule_totes(n_days=25):
    # constants

    #per unit costs
    storage_cost = 0.1
    labor_cost = 1.0
    labor_cost_extra_weekend = 0.5

    sales_base = 10.0
    inventory_max = 70.0
    inventory_start = 20.0
    production_max = 90.0
    sales_max = 100.0
    ramp = 1.0
    days_until_cleaning = 30

    #totes

    max_items_per_tote = 4
    n_washed_max = 5
    cost_washing_tote = 0.2
    n_totes_washed_start = 2


    # constant storage costs
    storage_costs = np.repeat(storage_cost, n_days)

    days = np.arange(n_days)
    weekday = days % 7

    # labor costs are higher on the weekend
    labor_costs = np.repeat(labor_cost, n_days) + (weekday > 4) * labor_cost_extra_weekend

    # sales vary by weekday and are ramping up
    weekday_sales = np.array([1.0, 1.5, 1.8, 1.6, 1.9, 2.7, 3.5])
    sales = np.repeat(sales_base, n_days) + weekday_sales[weekday] * ramp*days
    sales = np.array([min(sale, sales_max) for sale in sales])

    # define variables
    production = Variable(n_days)
    inventory = Variable(n_days)
    n_totes_washed = Variable(n_days)

    # Conservation of inventory equation
    D1 = first_deriv_matrix(n_days)
    difference = production - sales
    inventory_conservation = D1*inventory == difference[:-1]

    cum_matrix = get_step_function_matrix(n_days)
    cum_matrix_np = spmatrix2np(cum_matrix)

    shift_matrix = get_time_shift_matrix(n_days, days_until_cleaning)
    n_totes_become_dirty = (shift_matrix*n_totes_washed)[:n_days]

    n_washed_totes_available = n_totes_washed_start \
        + cum_matrix*n_totes_washed - n_totes_become_dirty


    objective = Minimize(production.T*labor_costs
                         + inventory.T*storage_costs
                         + cost_washing_tote*sum(n_totes_washed))

    constraints = [inventory >= 0,
                   inventory <= inventory_max,
                   production >= 0,
                   production <= production_max,
                   inventory_conservation,
                   inventory[0] == inventory_start,
                   n_totes_washed >= 0,
                   n_totes_washed <= n_washed_max,
                   inventory < max_items_per_tote*n_washed_totes_available]

    # define the problem and solve it

    problem = Problem(objective, constraints)
    problem.solve(verbose=True)

    print "Status: %s" % problem.status
    if problem.status == 'infeasible':
        return

    print 'plotting'
    #make plots
    plt.clf()
    plt.plot(days, production.value, label='production', marker='o')
    plt.plot(days, inventory.value, label='inventory')
    plt.plot(days, sales, label='sales', linestyle='--')
    plt.plot(days, n_washed_totes_available.value, label='clean totes', linestyle='--')
    plt.xlabel('Day')
    plt.title('Production Schedule: One product')
    plt.legend()
    plt.show()

    return problem

if __name__ == "__main__":
    print 'close window to finish'
    create_schedule_totes(50)
