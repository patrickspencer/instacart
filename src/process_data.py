# -*- coding: utf-8 -*-

"""
process_data.py
~~~~~~~~~~~~~~~

functions for processing the data before it goes into the model. This include
generating new features and dropping unused features.
"""


def fillna(df):
    df['days_since_prior_order'] = df['days_since_prior_order'].fillna(-1)
    df['add_to_cart_order'] = df['add_to_cart_order'].fillna(-1)
    return df


class ProcessedData:

    def __init__(self, df):
        df = fillna(df)
        dropped_columns = ['order_id', 'user_id', 'product_id']
        self.df = df.drop(dropped_columns, axis=1)
        self.train = self.df.loc[self.df['eval_set'] == 'train'].drop(['eval_set'], axis=1)
        self.test = self.df.loc[self.df['eval_set'] == 'test'].drop(['eval_set'], axis=1)
        self.X_train = self.train.drop(['reordered'], axis=1)
        self.X_test = self.test.drop(['reordered'], axis=1)
        self.y_train = self.train['reordered']
