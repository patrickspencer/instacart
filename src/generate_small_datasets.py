# -*- coding: utf-8 -*-

"""
generate_small_datasets.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

The tables are too large to handle using on a local machine so we are going to
take a random sample of the users and use those to construct the data pipelines
that create the dataframes and features that we will use to make the final
model. It's important to note that we should not expect a random sample to be a
good training set because it would not accurately represent the data. We are
just doing it to be able to test that our features are created correctly.
"""
import numpy as np
import pandas as pd

# first have to run generate_full_orders_df.py
# orders_full = pd.read_csv('~/instacart_data/orders_full.csv')
orders = pd.read_csv('~/instacart_data/orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

def sample_orders(classification='train', sample_size=100, seed=0):
    mask = orders['eval_set'] == classification
    return orders[mask]['user_id'].sample(sample_size, random_state=seed).values

if __name__ == '__main__':

    # Random sample and make a 70%, 30% split for train, validation, and test
    train_set_users = sample_orders('train', 100, 1)
    test_set_users = sample_orders('test', 30, 6)

    # check that none of the train, validation, or test sets intersect
    assert not set(train_set_users).intersection(set(test_set_users))

    # find all orders in orders_full whose users are in each set
    train_set = orders.loc[orders['user_id'].isin(train_set_users)]
    test_set = orders.loc[orders['user_id'].isin(test_set_users)]

    df_full = pd.concat([train_set, test_set])

    # save data
    data_dir = '../data'
    df_full.to_csv(data_dir + '/orders_small.csv', index=False)
