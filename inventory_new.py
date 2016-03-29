from cvxpy import Variable, Minimize, Problem, Bool, Int
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

    epsilon = 1e-9
    large = 1e9

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
    max_totes_washed_per_day = 100
    n_products_per_tote = 100
    cost_of_washing = 0.1

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

    # include nearly negligible price increase just to
    # ensure lower indices are used first
    cost_of_totes = cost_of_new_tote + epsilon*np.arange(n_totes)
    cost_of_washes = cost_of_washing + epsilon*np.arange(n_totes)

    # define variables
    production = Variable(n_days)
    inventory = Variable(n_days)
    tote_is_purchased = Bool(n_totes, n_days)
    tote_is_clean = Bool(n_totes, n_days)
    n_products_in_tote = Int(n_totes, n_days)

    # Conservation of inventory equation
    D1 = first_deriv_matrix(n_days)
    difference = production - sales
    inventory_conservation = D1*inventory == difference[:-1]

    unity_totes = np.repeat(1.0, n_totes)

    number_of_totes_each_day = tote_is_purchased.T*unity_totes
    number_of_clean_totes_each_day = (tote_is_clean.T*unity_totes)
    enough_clean_totes = inventory <= n_products_per_tote*number_of_clean_totes_each_day

    balance_inventory_in_totes = (n_products_in_tote.T*unity_totes) == inventory

    n_totes_does_not_decrease = D1 * number_of_totes_each_day >= 0
    n_clean_totes_does_not_decrease = D1 * number_of_clean_totes_each_day >= 0

    objective = Minimize(production.T*labor_costs
                         + inventory.T*storage_costs
                         + sum((D1*tote_is_purchased.T)*cost_of_totes)
                         + sum((D1*tote_is_clean.T)*cost_of_washes))

    constraints = [inventory >= 0,
                   inventory <= inventory_max,
                   production >= 0,
                   production <= production_max,
                   inventory_conservation,
                   inventory[0] == inventory_start,
                   tote_is_clean <= tote_is_purchased,
                   enough_clean_totes,
                   n_totes_does_not_decrease,
                   n_clean_totes_does_not_decrease,
                   number_of_totes_each_day[0] == 30,
                   number_of_clean_totes_each_day[0] == 30,
                   balance_inventory_in_totes,
                   D1*number_of_clean_totes_each_day <= max_totes_washed_per_day]

    # define the problem and solve it

    problem = Problem(objective, constraints)
    problem.solve(verbose=True)
    print 'production: ', production.value
    print 'inventory: ', inventory.value
    print 'totes is purchased: ', tote_is_purchased.value
    print 'tote is cleaned: ', tote_is_clean.value

    print "Status: %s" % problem.status
    if problem.status == 'infeasible':
        return

    print 'plotting'
    plt.clf()
    #make plots
    plt.plot(days, production.value, label='production', marker='o')
    plt.plot(days, inventory.value, label='inventory')
    plt.plot(days, sales, label='sales', linestyle='--')
    plt.xlabel('Day')
    plt.title('Production Schedule: One product')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print 'close window to finish'
    create_schedule(5)
