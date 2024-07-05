SELECT count(*) from item, store_sales, store_returns           where item.i_item_sk = store_sales.ss_item_sk AND item.i_item_sk = store_returns.sr_item_sk;
SELECT count(*) from item, store_sales, catalog_sales           where item.i_item_sk = store_sales.ss_item_sk AND item.i_item_sk = catalog_sales.cs_item_sk;
SELECT count(*) from item, store_returns,  catalog_sales        where item.i_item_sk = store_returns.sr_item_sk AND item.i_item_sk = catalog_sales.cs_item_sk;
SELECT count(*) from store_sales, store_returns, catalog_sales  where store_sales.ss_item_sk=store_returns.sr_item_sk AND store_sales.ss_item_sk=catalog_sales.cs_item_sk;
SELECT count(*) FROM customer, store_sales, store_returns       where customer.c_customer_sk=store_sales.ss_customer_sk AND customer.c_customer_sk=store_returns.sr_customer_sk;
SELECT count(*) FROM customer, store_sales, catalog_sales       where customer.c_customer_sk=store_sales.ss_customer_sk AND customer.c_customer_sk=catalog_sales.cs_bill_customer_sk;
SELECT count(*) FROM customer, store_returns, catalog_sales     where customer.c_customer_sk=store_returns.sr_customer_sk AND customer.c_customer_sk=catalog_sales.cs_bill_customer_sk;
SELECT count(*) FROM store_sales, store_returns, catalog_sales  where store_sales.ss_customer_sk=store_returns.sr_customer_sk AND store_sales.ss_customer_sk=catalog_sales.cs_bill_customer_sk;