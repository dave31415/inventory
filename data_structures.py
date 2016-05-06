# product schedule (can mean demand or production or whatever)

product_schedule = [{'period': '20160922',
                     'period_type': 'day',
                     'value': 4050}]

# demand keyed by product number

demand = {2622: product_schedule, 888: product_schedule}

# line schedule is the result

line_schedule = [{'line_number': 1,
                  'production_schedules':
                  {2622: product_schedule}}]

# This defines the thing that we are to optimize
# We either need to find a product which does this out of the box
# Or we need to create this


def cost_function(line_schedule, plant_parameters, constraints):
    """
    :param line_schedule: the line schedule
    :param plant_parameters: parameters defining the plant
    :param constraints: the constraints (besides inventory balance)
    :return: the cost, a number that we want to be small
    """
    pass


def minimize_cost_function(demand_schedule, plant_parameters, constraints):
    """
    Find the line schedule that minimizes the cost function within the
    bounds of the constraints
    :param plant_parameters: parameters defining the plant
    :param constraints: the constraints (besides inventory balance)
    :return: the best line_schedule
    """
    pass




