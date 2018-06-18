# -*- coding: utf-8 -*-

"""
process_data.py
~~~~~~~~~~~~~~~

functions for processing the data before it goes into the model. This include
generating new features and dropping unused features.
"""

def drop_columns(df):
    dropped_columns = ['order_id', 'user_id', 'eval_set', 'product_id']
    return df.drop(dropped_columns, axis=1)

def fillna(df):
    df['days_since_prior_order'] = df['days_since_prior_order'].fillna(-1)
    df['add_to_cart_order'] = df['add_to_cart_order'].fillna(-1)
    return df

def target_class(df, classification):
    return df.loc[df['eval_set'] == classification]

class ProcessedData:

    def __init__(self, df, classification):
        df = fillna(drop_columns(target_class(df, classification)))
        self.df = df
        self.X = df.drop(['reordered'], axis=1)
        self.y = df['reordered']
