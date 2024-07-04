from joins.schema_base import SchemaGraph, Table


def gen_tpcds_light_schema(hdf_path):
    schema = SchemaGraph()

    # item store_sales store_returns catalog_sales,customer
    # tables
    # item
    schema.add_table(
        Table(
            "item",
            primary_key=["i_item_sk"],
            attributes=["i_item_sk", "i_rec_start_date", "i_current_price"],
            irrelevant_attributes=[],  # ['Id'],
            no_compression=[],
            csv_file_location=hdf_path.format("item"),
            table_size=18000,
        )
    )

    # customer
    schema.add_table(
        Table(
            "customer",
            primary_key=["c_customer_sk"],
            attributes=[],
            irrelevant_attributes=[],  # ['Id'],
            no_compression=[],
            csv_file_location=hdf_path.format("customer"),
            table_size=100000,
        )
    )

    # store_sales
    schema.add_table(
        Table(
            "store_sales",
            primary_key=["ss_item_sk", "ss_customer_sk"],
            attributes=["ss_list_price", "ss_sales_price", "ss_net_profit"],
            irrelevant_attributes=[],  # ['Id'],
            no_compression=[],
            csv_file_location=hdf_path.format("store_sales"),
            table_size=2880404,
        )
    )

    # store_returns
    schema.add_table(
        Table(
            "store_returns",
            primary_key=["sr_item_sk", "sr_customer_sk"],
            attributes=["sr_return_quantity", "sr_return_tax", "sr_net_loss"],
            irrelevant_attributes=[],  # ['Id'],
            no_compression=[],
            csv_file_location=hdf_path.format("store_returns"),
            table_size=287514,
        )
    )

    # catalog_sales
    schema.add_table(
        Table(
            "catalog_sales",
            primary_key=["cs_item_sk", "cs_customer_sk"],
            attributes=["cs_quantity", "cs_sales_price", "cs_net_profit"],
            irrelevant_attributes=[],  # ['Id'],
            no_compression=[],
            csv_file_location=hdf_path.format("catalog_sales"),
            table_size=1441548,
        )
    )

    # item store_sales store_returns catalog_sales,customer
    schema.add_pk_bins("item", "i_item_sk", 1, 18000)
    schema.add_pk_bins("store_sales", "ss_item_sk", 1, 18000)
    schema.add_pk_bins("store_returns", "sr_item_sk", 1, 18000)
    schema.add_pk_bins("catalog_sales", "cs_item_sk", 1, 18000)

    schema.add_pk_bins("customer", "c_customer_sk", 1, 100000)
    schema.add_pk_bins("store_sales", "ss_customer_sk", 1, 100000)
    schema.add_pk_bins("store_returns", "sr_customer_sk", 1, 100000)
    schema.add_pk_bins("catalog_sales", "cs_customer_sk", 1, 100000)

    schema.add_join_path(
        "SELECT count(*) from item, store_sales,  store_returns,  catalog_sales where i_item_sk = ss_item_sk AND i_item_sk=sr_item_sk AND i_item_sk=cs_item_sk "
    )

    schema.add_join_path(
        "SELECT count(*) FROM customer, store_sales, store_returns, catalog_sales where c_customer_sk=ss_customer_sk AND c_customer_sk=sr_customer_sk AND c_customer_sk=cs_bill_customer_sk"
    )

    return schema
