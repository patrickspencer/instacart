# -*- coding: utf-8 -*-

"""
generate_full_orders.py
~~~~~~~~~~~~~~~~~~~~~~~

The `orders` table does not include information about the individual products
that go into each order. `orders` records the relationship between a user and
the order.

The tables `orders_prior` and `orders_train` relate the individual products
(represented by product ids) to the orders. These two tables also record the
order in which the item was added to the order and it this item is a reorder.
These tables have the following features:

- *order_id*
- *product_id*
- *add_to_cart_order*: what order the item was added to the cart. Was it the
  first added? The last?
- *reordered*: Has this person ordered this item in the past?

In order to get a large dataframe which includes all the information about the
individual orders and the order itself we would have to concatenate the
`orders_prior` and the `orders_train` table and then merge the new table with
the orders table on the `order_id` column. We want to include all the orders,
even if there are not products in the order, so we would left join the
`orders_prior + orders_train` on the `orders` table. This script does
the left join.

This take a while and produces a file called orders_full.csv which is about 1.9
gb.
"""
import pandas as pd

if __name__ == '__main__':

    data_dir = '~/instacart_data/'
    orders = pd.read_csv(data_dir + 'orders.csv')
    orders_prior = pd.read_csv(data_dir + 'order_products__prior.csv')
    orders_train = pd.read_csv(data_dir + 'order_products__train.csv')

    df = pd.concat((orders_train, orders_prior), axis=0)
    df = orders.merge(df, on='order_id', how='left')

    # 'Unnamed: 0' column is the same as the index so it's redundant
    # df.drop(['Unnamed: 0'], axis=1, inplace=True)

    df.to_csv(data_dir + 'orders_full.csv', index=False)
