## Generate full dataframe of orders and users

`generate_full_orders_df.py`

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
`orders_prior + orders_train` on the `orders` table. This does the left join.

This take a while and produces a file called orders_full.csv which is about 1.9
gb.

## Random sample of users

`generate_small_datasets.py`

The tables are too large to handle using on a local machine so we are
going to take a random sample of the users and use those to construct
the data pipelines that create the dataframes and features that we will
use to make the final model. It's important to note that we should not
expect a random sample to be a good training set because it would not
accurately represent the data. We are just doing it to be able to test
that our features are created correctly. The file
`generate_small_datasets.py` generates the small test sets.
