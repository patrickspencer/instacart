import sys
import numpy as np
import pandas as pd
sys.path.append('../src/')
from model import Model
from process_data import ProcessedData
from sklearn.linear_model import LogisticRegression


data_folder = '~/instacart_data/'
aisles = pd.read_csv(data_folder + 'aisles.csv')
departments = pd.read_csv(data_folder + 'departments.csv')

print('loading products')
products = pd.read_csv(data_folder+ 'products.csv', dtype={
        'product_id': np.uint16,
        'order_id': np.int32,
        'aisle_id': np.uint8,
        'department_id': np.uint8},
        usecols=['product_id', 'aisle_id', 'department_id'])

print('loading orders')
# replace orders_full.csv with orders_full_small.csv for faster loading time
orders = pd.read_csv(data_folder + 'orders_full_small.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32,
        'product_id': np.uint16,
        'add_to_cart_order': np.int16,
        'reordered': np.int8})

orders = pd.read_csv('../data/orders_full_small.csv')
orders = ProcessedData(orders)
clf = LogisticRegression()
model = Model(clf, orders.X_train, orders.y_train, orders.X_test)
print(model.classification_report)
