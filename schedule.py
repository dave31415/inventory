"""
Create the production schedule optimization problem
"""
from cvxpy import Problem, Maximize, Variable
from plant_config import plant
from demand import create_demand_schedule


def create_variables(plant, demand_schedule):
    products = plant['products']
    product_names = [p['product_name'] for p in products]
    production_lines = plant['production_lines']
    n_days = len(demand_schedule['days'])

    production_variables = {k: {} for k in product_names}
    sales_variables = {}
    inventory_variables = {}

    for product_name in product_names:
        sales_variables[product_name] = Variable(n_days, name="sales: %s" % product_name)
        inventory_variables[product_name] = Variable(n_days,
                                                     name="inventory: %s" % product_name)
        for production_line in production_lines:
            pl_name = production_line['production_line_name']
            pl_costs = production_line['product_run_rate_costs']
            if product_name in pl_costs:
                print "Line: %s can make product: %s" % (pl_name, product_name)
                name = 'production: %s|%s' % (product_name, pl_name)
                production_variables[product_name][pl_name] = Variable(n_days, name=name)

    variables = {'production_variables': production_variables,
                 'sales_variables': sales_variables,
                 'inventory_variables': inventory_variables}
    return variables


def create_objective(variables, demand_schedule):
    return 1


def create_constraints(variables, plant, demand_schedule):
    return 1


def define_schedule_problem():
    """
    Define the optimization problem
    :return:
    """
    demand_schedule = create_demand_schedule()
    variables = create_variables(plant, demand_schedule)
    objective = create_objective(variables, demand_schedule)
    constraints = create_constraints(variables, plant, demand_schedule)
    problem = Problem(objective, constraints)
    return problem


def do_stuff_with_solution(problem):
    pass


def solve_problem(problem):
    problem.solve(verbose=True, solver='cvxpy')
    do_stuff_with_solution(problem)


