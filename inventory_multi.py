"""
Code for demonstrating the optimization of multiple
product supply chain schedule with variable labor charges,
multiple production lines, inventory storage
charges and the concept of totes.
Totes are boxes holding a maximum number of products
and require cleaning every 30 days at some cost with
a maximum number of totes that can be cleaned in a single day.
"""

from cvxpy import Variable, Minimize, Problem, sum_entries
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
    Warning: if you change these it may result
    in an infeasible solution so choose them wisely
    :return: dictionary of all parameters needed
    """
    params = {'n_days': 80,
              'labor_cost': 1.0,
              'labor_cost_extra_weekend': 0.5,
              'storage_cost': 0.1,
              'washing_tote_cost': 8.9,
              'sales_base': 10.0,
              'sales_ramp': 1.0,
              'sales_random_seed': 42,
              'sales_sigma': 12.0,
              'sales_max': 110.0,
              'inventory_start': 20.0,
              'inventory_max': 1000.0,
              'production_max': 1005.0,
              'days_until_cleaning': 30,
              'max_items_per_tote': 4,
              'max_washed_per_day': 5,
              'n_totes_washed_start': 100,
              'production_cost': 1.0}

    return params


def create_product_line_costs():
    """
    Create a list of lists to keep track of the
    extra cost associated with creating a particular
    product on a particular product line.
    Example:
    costs = create_product_line_params():
    # cost of product i on product line j is
    costs[i][j]
    :return: list of lists
    """

    # rather than say that some products can't be made on
    # some product lines, make it possible but only
    # with enormous cost

    large_number = 10000.0

    # 3 product lines and 5 products

    product_costs_per_line = [np.array([1.0, 2.5, large_number, 1.0, 4.0]),
                              np.array([2.0, 0.8, 3.0, 1.0, large_number]),
                              np.array([large_number, 12.0, 2.4, 1.5, 2.5])]

    n_products = len(product_costs_per_line[0])
    for costs in product_costs_per_line:
        assert len(costs) == n_products

    return product_costs_per_line


def create_sales(days, pars, product_num=0):
    """
    Create simulated sales
    that are ramping up but saturate and have weekday
    dependence and some (pseudo) randomness on top
    :param days: index of days
    :param pars: parameters, e.g. from create_default_params
    :product_num: product number, makes different sales for each
    :return: numpy array of sales for each day
    """
    weekday = days % 7
    weekday_sales = np.array([1.0, 1.5, 1.8, 1.6, 1.9, 2.7, 3.5])
    sales = np.repeat(pars['sales_base'] + 4.0*product_num, pars['n_days']) \
        + weekday_sales[weekday] * pars['sales_ramp']*days/(1.0 + product_num)
    sales = np.array([min(sale, pars['sales_max']) for sale in sales])
    np.random.seed(pars['sales_random_seed'] + product_num)
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

    product_costs = create_product_line_costs()
    n_product_lines, n_products, = len(product_costs), len(product_costs[0])

    days = np.arange(pars['n_days'])

    sales = np.zeros((pars['n_days'], n_products))
    for product_num in xrange(n_products):
        sales[:, product_num] = create_sales(days, pars, product_num)

    labor_costs = get_labor_costs(days, pars)

    # define variables which keep track of production,
    # inventory and number of totes washed per day

    # defines productions as a list of 2D variables because
    # cvxpy does not handle higher than 2D

    productions = [Variable(pars['n_days'], n_products)]*n_product_lines
    inventory = Variable(pars['n_days'], n_products)
    n_totes_washed = Variable(pars['n_days'])

    # calculate when the totes that were washed become dirty again
    shift_matrix = mu.time_shift_matrix(pars['n_days'],
                                        pars['days_until_cleaning'])

    n_totes_become_dirty = (shift_matrix*n_totes_washed)[:pars['n_days']]

    # calculate the number of clean totes on any day
    cum_matrix = mu.cumulative_matrix(pars['n_days'])

    n_washed_totes_available = pars['n_totes_washed_start'] \
        + cum_matrix*(n_totes_washed - n_totes_become_dirty)

    # Minimize total cost which is sum of labor costs,
    # storage costs, washing costs and other cost
    # of producing each product on each product line

    all_costs = pars['storage_cost'] * sum_entries(inventory) + \
        pars['washing_tote_cost'] * sum_entries(n_totes_washed)

    for i, production in enumerate(productions):
        all_costs += pars['production_cost'] * sum_entries(production*product_costs[i])
        all_costs += sum_entries(production.T*labor_costs)
    objective = Minimize(all_costs)

    # Subject to these constraints

    # Inventory continuity equation
    derivative_matrix = mu.first_deriv_matrix(pars['n_days'])

    production_all_lines = sum(productions)
    difference = production_all_lines - sales
    inventory_continuity = derivative_matrix * inventory == difference[:-1, :]

    # Have enough clean totes to hold all the inventory

    production_all_products = sum_entries(production_all_lines, 1)
    products_owned = sum_entries(inventory, 1) + production_all_products
    have_enough_clean_totes = \
        products_owned <= pars['max_items_per_tote']*n_washed_totes_available

    constraints = [inventory >= 0,
                   inventory <= pars['inventory_max'],
                   inventory_continuity,
                   inventory[0, :] == pars['inventory_start'],
                   n_totes_washed >= 0,
                   n_totes_washed <= pars['max_washed_per_day'],
                   have_enough_clean_totes]

    for production in productions:
        constraints.append(production >= 0)
        constraints.append(production <= pars['production_max'])

    # define the problem and solve it

    problem = Problem(objective, constraints)
    problem.solve(verbose=True)

    print "Status: %s" % problem.status
    if problem.status == 'infeasible':
        print "Problem is infeasible, no solution found"
        return

    total_cost = problem.value
    total_washing_cost = pars['washing_tote_cost']*sum_entries(n_totes_washed).value
    total_labor_cost = 0.0
    for production in productions:
        total_labor_cost += sum_entries(production.T*labor_costs).value
    total_storage_cost = sum_entries(inventory).value*pars['storage_cost']

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
        plt.plot(days, n_washed_totes_available.value,
                 label='clean totes', linestyle='--')
        plt.xlabel('Day')
        plt.title('Production Schedule: One product with totes')
        plt.legend()
        plt.xlim(-1, pars['n_days'] + 15)
        y_max = 9 + max([max(production.value),
                         max(inventory.value),
                         max(sales)])
        plt.ylim(-2, y_max)
        plt.show()


if __name__ == "__main__":
    print 'close window to finish'
    create_schedule()
