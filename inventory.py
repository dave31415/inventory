"""
Code for demonstrating the optimization of a single
product supply chain schedule with variable labor charges,
inventory storage charges and the concept of totes.
Totes are boxes holding a maximum number of products
and require cleaning every 30 days at some cost with
a maximum number of totes that can be cleaned in a single day.
"""

from cvxpy import Variable, Maximize, Problem
import numpy as np
import matrix_utils as mu
from matplotlib import pylab as plt
from time import time
import epopt


try:
    import seaborn
except ImportError:
    print 'seaborn not available'
    pass


def create_default_params():
    """
    Create the default parameters for the demo
    Warning: if you change these it may result
    in an infeasible solution so choose them wisely
    :return: dictionary of all parameters needed
    """
    params = {'n_days': 80,
              'labor_cost': 1.0,
              'labor_cost_extra_weekend': 0.5,
              'storage_cost': 0.05,
              'washing_tote_cost': 18.9,
              'demand_base': 10.0,
              'demand_ramp': 1.0,
              'demand_random_seed': 42,
              'demand_sigma': 17.0,
              'demand_max': 110.0,
              'inventory_start': 20.0,
              'inventory_max': 100.0,
              'production_max': 157.0,
              'days_until_cleaning': 30,
              'max_items_per_tote': 4,
              'max_washed_per_day': 5,
              'n_totes_washed_start': 1,
              'sales_price': 1.60,
              'use_epsilon_solver': False,
              'verbose': True}

    return params


def create_demand(days, pars):
    """
    Create simulated sales demand
    that are ramping up but saturate and have weekday
    dependence and some (pseudo) randomness on top
    :param days: index of days
    :param pars: parameters, e.g. from create_default_params
    :return: numpy array of sales for each day
    """
    weekday = days % 7
    weekday_demand = np.array([1.0, 1.5, 1.8, 1.6, 1.9, 2.7, 3.5])
    demand = np.repeat(pars['demand_base'], pars['n_days']) \
        + weekday_demand[weekday] * pars['demand_ramp']*days
    demand = np.array([min(sale, pars['demand_max']) for sale in demand])
    np.random.seed(pars['demand_random_seed'])
    demand += np.random.randn(pars['n_days'])*pars['demand_sigma']
    demand += 70.5*(np.sin(2*np.pi*days/365.0)**2)

    demand[demand < 0] = 0.0
    return demand


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


def create_schedule(pars=None, do_plot=True):
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

    print 'creating demand'
    demand = create_demand(days, pars)
    labor_costs = get_labor_costs(days, pars)

    # define variables which keep track of
    # production, inventory and number of totes washed per day

    print 'defining variables'
    production = Variable(pars['n_days'])
    sales = Variable(pars['n_days'])
    inventory = Variable(pars['n_days'])
    n_totes_washed = Variable(pars['n_days'])

    print 'calculating costs and profit'
    # calculate when the totes that were washed become dirty again
    shift_matrix = mu.time_shift_matrix(pars['n_days'],
                                            pars['days_until_cleaning'])

    n_totes_become_dirty = (shift_matrix*n_totes_washed)[:pars['n_days']]

    # calculate the number of clean totes on any day
    cum_matrix = mu.cumulative_matrix(pars['n_days'])

    n_washed_totes_available = pars['n_totes_washed_start'] \
        + cum_matrix*(n_totes_washed - n_totes_become_dirty)

    print 'calculating total cost'

    # Minimize total cost which is
    # sum of labor costs, storage costs and washing costs

    total_cost = production.T*labor_costs + \
                 pars['storage_cost'] * sum(inventory) + \
                 pars['washing_tote_cost'] * sum(n_totes_washed)

    total_profit = pars['sales_price']*sum(sales)-total_cost

    print 'defining objective'
    objective = Maximize(total_profit)

    # Subject to these constraints

    # Inventory continuity equation
    derivative_matrix = mu.first_deriv_matrix(pars['n_days'])
    difference = production - sales
    inventory_continuity = derivative_matrix * inventory == difference[:-1]

    # Have enough clean totes to hold all the inventory

    products_owned = inventory + production
    have_enough_clean_totes = \
        products_owned <= pars['max_items_per_tote']*n_washed_totes_available

    constraints = [inventory >= 0,
                   inventory <= pars['inventory_max'],
                   production >= 0,
                   production <= pars['production_max'],
                   inventory_continuity,
                   inventory[0] == pars['inventory_start'],
                   n_totes_washed >= 0,
                   n_totes_washed <= pars['max_washed_per_day'],
                   have_enough_clean_totes,
                   sales > 0,
                   sales <= demand]

    # define the problem and solve it

    problem = Problem(objective, constraints)

    solver = 'cvxpy'
    if pars['use_epsilon_solver']:
        solver = 'epsilon'

    print 'solving with: %s' % solver
    start = time()
    if not pars['use_epsilon_solver']:
        problem.solve(verbose=pars['verbose'])
    else:
        epopt.solve(problem, verbose=pars['verbose'])

    finish = time()
    run_time = finish - start
    print 'Solve time: %s seconds' % run_time

    print "Status: %s" % problem.status
    if problem.status == 'infeasible':
        print "Problem is infeasible, no solution found"
        return

    n_items_sold = sum(sales.value)
    total_cost = problem.value
    total_washing_cost = pars['washing_tote_cost']*sum(n_totes_washed.value)
    total_labor_cost = (production.T*labor_costs).value
    total_storage_cost = sum(inventory.value)*pars['storage_cost']
    total_cost_per_item = problem.value/n_items_sold

    print "Total cost: %s" % total_cost
    print "Total labor cost: %s" % total_labor_cost
    print "Total washing cost: %s" % total_washing_cost
    print "Total storage cost: %s" % total_storage_cost
    print "Total cost/item: %s" % total_cost_per_item
    print "Total profit: %s" % total_profit.value

    if do_plot:
        plt.clf()
        plt.plot(days, production.value, label='production', marker='o')
        plt.plot(days, inventory.value, label='inventory')
        plt.plot(days, sales.value, label='sales', linestyle='-')
        plt.plot(days, demand, label='demand', linestyle='--')
        plt.plot(days, n_washed_totes_available.value,
                 label='clean totes', linestyle='--')
        plt.xlabel('Day')
        plt.title('Production Schedule: One product with totes')
        plt.legend()
        plt.xlim(-1, pars['n_days'] + 15)
        y_max = 9 + max([max(production.value),
                         max(inventory.value),
                         max(demand)])
        plt.ylim(-2, y_max)
        plt.show()


def run():
    """
    Run the pipeline
    :return: None
    """
    print 'close window to finish'
    start = time()
    pars = create_default_params()
    pars['n_days'] = 500
    create_schedule(pars=pars, do_plot=False)
    finish = time()
    run_time = finish - start
    print 'total time: %s seconds' % run_time


def epsilon_failing():
    pars = create_default_params()
    pars['use_epsilon_solver'] = False
    create_schedule(pars=pars, do_plot=False)
    print 'cxvpy succeeded'
    pars['use_epsilon_solver'] = True
    create_schedule(pars=pars, do_plot=False)
    print 'epsilon succeeded'


if __name__ == "__main__":
    run()
    # epsilon_failing()