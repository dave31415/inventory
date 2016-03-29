from cvxpy import Variable, Minimize, Problem
import numpy as np
import scipy
import cvxopt
from matplotlib import pylab as plt

# one product, simplest supply chain


def first_deriv_matrix(n):
    # a matrix which computes the first derivative of a vector
    # copied from examples on cvxpy page
    e = np.mat(np.ones((1, n)))
    D = scipy.sparse.spdiags(np.vstack((-e, e)), range(2), n-1, n)
    D_coo = D.tocoo()
    D = cvxopt.spmatrix(D_coo.data, D_coo.row.tolist(), D_coo.col.tolist())
    return D


def create_schedule(n_days=25):
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

    #totes

    # total number of totes that exist in the world
    # and can be purchased, cannot sell totes once purchased
    n_totes = 100
    cost_of_new_tote = 0.2
    cost_of_storing_tote = 0.02
    max_totes_washed_per_day = 5

    weekday_sales = np.array([1.0, 1.5, 1.8, 1.6, 1.9, 2.7, 3.5])

    # constant storage costs
    storage_costs = np.repeat(storage_cost, n_days)

    days = np.arange(n_days)
    weekday = days % 7

    # labor costs are higher on the weekend
    labor_costs = np.repeat(labor_cost, n_days) + (weekday > 4) * labor_cost_extra_weekend

    # sales vary by weekday and are ramping up
    sales = np.repeat(sales_base, n_days) + weekday_sales[weekday] * ramp*days
    sales = np.array([min(sale, sales_max) for sale in sales])

    # define variables
    production = Variable(n_days)
    inventory = Variable(n_days)
    tote_is_purchased = Variable(n_totes, n_days)

    objective = Minimize(production.T*labor_costs
                         + inventory.T*storage_costs
                         + totes.T*cost_of_storing_tote)

    # Conservation of inventory equation
    D1 = first_deriv_matrix(n_days)
    difference = production - sales
    inventory_conservation = D1*inventory == difference[:-1]

    constraints = [inventory >= 0,
                   inventory <= inventory_max,
                   production >= 0,
                   production <= production_max,
                   inventory_conservation,
                   inventory[0] == inventory_start]

    # define the problem and solve it

    problem = Problem(objective, constraints)
    problem.solve()

    plt.clf()
    print "Status: %s" % problem.status
    if problem.status == 'infeasible':
        return


    #make plots
    plt.plot(days, production.value, label='production', marker='o')
    plt.plot(days, inventory.value, label='inventory')
    plt.plot(days, sales, label='sales', linestyle='--')
    plt.xlabel('Day')
    plt.title('Production Schedule: One product')
    plt.legend()




