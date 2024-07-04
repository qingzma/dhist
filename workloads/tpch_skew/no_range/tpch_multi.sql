SELECT count(*)  FROM customer, orders, lineitem WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey;

SELECT count(*)  FROM nation, supplier, customer, orders, lineitem WHERE n_nationkey = s_nationkey AND s_nationkey = c_nationkey AND c_custkey = o_custkey AND o_orderkey = l_orderkey;

nation key: nation - customer -supplier 

query 17,29:
item_sk: item store_sales store_returns catalog_sales
SELECT count(*) from item, store_sales,  store_returns,  catalog_sales 
where i_item_sk = ss_item_sk AND i_item_sk=sr_item_sk AND i_item_sk=cs_item_sk 

query 25
customer sk: 
SELECT count(*) FROM store_sales, store_returns, catalog_sales, 
? where ss_customer_sk AND sr_customer_sk AND cs_bill_customer_sk