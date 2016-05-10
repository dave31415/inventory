# define the plant which involves cost structure
# production lines, products and extra costs associated with producing
# each product on each production line

plant_params = {'labor_cost': 1.0,
                'labor_cost_extra_weekend': 0.5,
                'storage_cost': 0.05,
                'inventory_max': 100.0,
                'production_max': 157.0,
                'sales_price': 1.60}

products = [{'product_name': 'hand_soap', 'sales_price': 1.0},
            {'product_name': 'laundry_detergent', 'sales_price': 5.0},
            {'product_name': 'shampoo', 'sales_price': 2.4}]

production_lines = [{'production_line_name': 'Line_1',
                     'employees_required': 15,
                     'product_run_rate_costs': {'hand_soap': 1.0,
                                                'laundry_detergent': 5.0,
                                                'shampoo': 7.0}},
                    {'production_line_name': 'Line_2',
                     'employees_required': 23,
                     'product_run_rate_costs': {'hand_soap': 1.0,
                                                'laundry_detergent': 4.0}},
                    {'production_line_name': 'Line_3',
                     'employees_required': 12,
                     'product_run_rate_costs': {'hand_soap': 0.1,
                                                'laundry_detergent': 105.0}}]

plant = {'params': plant_params,
         'products': products,
         'production_lines': production_lines}


def assert_production_line_products_in_product_list(production_lines, products):
    product_names = [product['product_name'] for product in products]
    n_bad = 0
    for production_line in production_lines:
        for product in production_line['product_run_rate_costs'].keys():
            if product not in product_names:
                print 'product: %s not in line %s' % \
                      (product, production_line['production_line_name'])
                n_bad += 1

    if n_bad != 0:
        raise ValueError('Some products in product lines that are not in product list')

assert_production_line_products_in_product_list(production_lines, products)
