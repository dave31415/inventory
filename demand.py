import numpy as np
import plant

demand_params = {'n_days': 200,
                 'demand_base': 10.0,
                 'demand_ramp': 1.0,
                 'demand_random_seed': 42,
                 'demand_sigma': 17.0,
                 'demand_max': 110.0}


def create_demand(days, product_index=0, pars=None):
    """
    Create simulated sales demand
    that are ramping up but saturate and have weekday
    dependence and some (pseudo) randomness on top
    :param days: index of days
    :param pars: parameters, e.g. from create_default_params
    :param product_index: integer, default zero, changes the random
           seed and also the other curve shape parameters, so this can
           be run to create some slightly different curves for different \
           products
    :return: numpy array of sales for each day
    """
    if pars is None:
        pars = demand_params
    n_days = len(days)
    weekday = days % 7
    weekday_demand = np.array([1.0, 1.5, 1.8, 1.6, 1.9, 2.7, 3.5])
    demand = np.repeat(pars['demand_base'] + product_index, n_days) \
        + weekday_demand[weekday] * (pars['demand_ramp']*(5.0 / 5.0 + product_index))*days
    demand = np.array([min(sale, pars['demand_max']) for sale in demand])
    np.random.seed(pars['demand_random_seed'] + product_index)
    demand += np.random.randn(n_days)*pars['demand_sigma']
    demand += 70.5*(np.sin(2*np.pi*days/365.0)**2)

    demand[demand < 0] = 0.0
    return demand


def create_demand_multiple_products(days, product_names, pars=None):
    """
    Create simulated sales demand
    that are ramping up but saturate and have weekday
    dependence and some (pseudo) randomness on top
    This creates demand schedules for multiple products
    :param days: index of days
    :param pars: parameters, e.g. from create_default_params
    :return: dict of numpy arrays of sales for each day,
             keyed by product name
    """
    return {product: create_demand(days, product_index=i, pars=pars)
            for i, product in enumerate(product_names)}


def create_demand_schedule(pars=None):
    """
    Create the demand schedule for products defined in plant module
    :param pars: demand_pars, uses global default by default
    :return: a demand schedule
    """
    if pars is None:
        pars = demand_params
    product_names = [product['product_name'] for product in plant.products]
    days = np.arange(pars['n_days'])
    demand = create_demand_multiple_products(days, product_names, pars=pars)
    demand_schedule = {'days': days,
                       'demand': demand}
    return demand_schedule