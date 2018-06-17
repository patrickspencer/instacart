# Instacart Kaggle Exploration

## Overview

Original Kaggle description is here:

[https://www.kaggle.com/c/instacart-market-basket-analysis#](https://www.kaggle.com/c/instacart-market-basket-analysis#)

## Data

## The three 'orders' tables

There are three sets of data which have information about the orders that customers have made.

**Orders**: This table includes information about all the orders placed. The features are:
- *order_id*
- *user_id*
- *eval_set*: this feature tells which of the three sets the order belongs in (prior, train, test)
- *order_number*
- *order_dow*: not sure what this feature is
- *order_hour_of_the_day*: hour of the day the order was places (out of a 24 hour clock)
- *days_since_prior_order*

The `orders` table does not include information about the individual products that go into each order. `orders` records the relationship between a user and the order. 

The tables `orders_prior` and `orders_train` relate the individual products (represented by product ids) to the orders. These two tables also record the order in which the item was added to the order and it this item is a reorder. These tables have the following features:

- *order_id*
- *product_id*
- *add_to_cart_order*: what order the item was added to the cart. Was it the first added? The last?
- *reordered*: Has this person ordered this item in the past?

In order to get a large dataframe which includes all the information
about the individual orders and the order itself we would have to
concatenate the `orders_prior` and the `orders_train` table and then
merge the new table with the orders table on the `order_id` column. We
want to include all the orders, even if there are not products in the
order so we would left join the `orders_prior + orders_train` on the
`orders` table which would look something like this:

```python
df = pd.concat((orders_train, orders_prior), axis=0)
df = orders.merge(df, on='order_id', how='left')
```

Notice that every order can have multiple items associated to it. For
example, take order_id 2539329. This is an order labeled as "prior" and
associated to user_id 1.

The file `src/generate_full_orders.py` does the concatenation and the
joining to produce a file called `orders_full.csv`.

## Files

### src/

Contains all models and python scripts.

`src/generate_full_orders.py`: Concatenates `orders_prior.csv` and
`orders_train.csv` and then joins the concatenation with `orders.csv`.
See `src/README.md` for more info.

`src/generate_small_datasets.py`: Samples `orders_full.csv` for smaller
train, validation, and test sets so we can test out our models and
feature engineering without dealing with a large file.
