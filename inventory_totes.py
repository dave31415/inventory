"""
Code for demonstrating the optimization of a single
product supply chain schedule with variable labor charges,
inventory storage charges and the concept of totes.
Totes are boxes holding a maximum number of products
and require cleaning every 30 days at some cost with
a maximum number of totes that can be cleaned in a single day.
"""

from cvxpy import Variable, Minimize, Problem
import numpy as np
import matrix_utils as mu
from matplotlib import pylab as plt

try:
    import seaborn
except ImportError:
    print 'seaborn not available'
    pass


def create_default_params():
    """
    Create the default parameters for the demo
    :return: dictionary of all parameters needed
    """
    params = {'n_days': 80,
              'labor_cost': 1.0,
              'labor_cost_extra_weekend': 0.5,
              'storage_cost': 0.1,
              'washing_tote_cost': 8.9,
              'sales_base': 10.0,
              'sales_random_seed': 42,
              'sales_sigma': 12.0,
              'inventory_max': 100.0,
              'inventory_start': 20.0,
              'production_max': 105.0,
              'sales_max': 110.0,
              'ramp': 1.0,
              'days_until_cleaning': 30,
              'max_items_per_tote': 4,
              'n_washed_max': 5,
              'n_totes_washed_start': 2}

    return params


def create_sales(days, pars):
    """
    Create simulated sales
    that are ramping up but saturate and have weekday
    dependence and some (pseudo) randomness on top
    :param days: index of days
    :param pars: parameters, e.g. from create_default_params
    :return: numpy array of sales for each day
    """
    weekday = days % 7
    weekday_sales = np.array([1.0, 1.5, 1.8, 1.6, 1.9, 2.7, 3.5])
    sales = np.repeat(pars['sales_base'], pars['n_days']) \
        + weekday_sales[weekday] * pars['ramp']*days
    sales = np.array([min(sale, pars['sales_max']) for sale in sales])
    np.random.seed(pars['sales_random_seed'])
    sales += np.random.randn(pars['n_days'])*pars['sales_sigma']
    sales[sales < 0] = 0.0
    return sales


def get_labor_costs(days, pars):
    """
    Return labor costs per day which are higher on the weekend
    :param days: numpy array of days
    :param pars: parameters
    :return: numpy array of labor costs per day
    """
    # labor costs are higher on the weekend
    weekday = days % 7
    labor_costs = np.repeat(pars['labor_cost'], pars['n_days']) \
        + (weekday > 4) * pars['labor_cost_extra_weekend']
    return labor_costs


def create_schedule_totes(pars=None, do_plot=True):
    """
    Demo an optimal supply chain scheduling with variable
    labor costs, and the concept of totes that hold a number of
    products. Totes need to be cleaned on a regular basis.
    :param pars: parameters from create_default_params
    :param do_plot: True if you want a plot created (default)
    :return: None
    """
    if pars is None:
        pars = create_default_params()

    days = np.arange(pars['n_days'])

    sales = create_sales(days, pars)
    labor_costs = get_labor_costs(days, pars)

    # define variables which keep track of
    # production, inventory and number of totes washed per day

    production = Variable(pars['n_days'])
    inventory = Variable(pars['n_days'])
    n_totes_washed = Variable(pars['n_days'])

    # TODO: check these next few parts for off-by-one errors
    # TODO: and that they are defined properly

    # calculate when the totes that were washed become dirty again
    shift_matrix = mu.get_time_shift_matrix(pars['n_days'],
                                            pars['days_until_cleaning'])

    n_totes_become_dirty = (shift_matrix*n_totes_washed)[:pars['n_days']]

    # calculate the number of clean totes on any day
    cum_matrix = mu.get_step_function_matrix(pars['n_days'])

    n_washed_totes_available = pars['n_totes_washed_start'] \
        + cum_matrix*(n_totes_washed - n_totes_become_dirty)

    # Minimize total cost
    # sum of labor costs, storage costs and washing costs

    objective = Minimize(production.T*labor_costs
                         + pars['storage_cost']*sum(inventory)
                         + pars['washing_tote_cost']*sum(n_totes_washed))

    # Subject to these constraints

    # Inventory continuity equation
    derivative_matrix = mu.first_deriv_matrix(pars['n_days'])
    difference = production - sales
    inventory_conservation = derivative_matrix * inventory == difference[:-1]

    # Have enough clean totes to hold all the inventory

    products_owned = inventory + production
    have_enough_clean_totes = \
        products_owned <= pars['max_items_per_tote']*n_washed_totes_available

    constraints = [inventory >= 0,
                   inventory <= pars['inventory_max'],
                   production >= 0,
                   production <= pars['production_max'],
                   inventory_conservation,
                   inventory[0] == pars['inventory_start'],
                   n_totes_washed >= 0,
                   n_totes_washed <= pars['n_washed_max'],
                   have_enough_clean_totes]

    # define the problem and solve it

    problem = Problem(objective, constraints)
    problem.solve(verbose=True)

    print "Status: %s" % problem.status
    if problem.status == 'infeasible':
        print "Problem is infeasible, no solution found"
        return

    total_cost = problem.value
    total_washing_cost = pars['washing_tote_cost']*sum(n_totes_washed.value)
    total_labor_cost = (production.T*labor_costs).value
    total_storage_cost = sum(inventory.value)*pars['storage_cost']

    print "Total cost: %s" % total_cost
    print "Total labor cost: %s" % total_labor_cost
    print "Total washing cost: %s" % total_washing_cost
    print "Total storage cost: %s" % total_storage_cost

    # sanity check
    sub_total = total_labor_cost + total_washing_cost + total_storage_cost
    assert abs(total_cost - sub_total) < 1e-5

    if do_plot:
        plt.clf()
        plt.plot(days, production.value, label='production', marker='o')
        plt.plot(days, inventory.value, label='inventory')
        plt.plot(days, sales, label='sales', linestyle='--')
        plt.plot(days, n_washed_totes_available.value, label='clean totes', linestyle='--')
        plt.xlabel('Day')
        plt.title('Production Schedule: One product with totes')
        plt.legend()
        plt.xlim(-1, pars['n_days'] + 15)
        y_max = 9 + max([max(production.value), max(inventory.value), max(sales)])
        plt.ylim(-2, y_max)
        plt.show()


if __name__ == "__main__":
    print 'close window to finish'
    create_schedule_totes()
