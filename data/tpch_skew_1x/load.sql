\copy region    from '/data1/tpch/skew/1/region.csv'     ( FORMAT CSV, DELIMITER('|') );
\copy nation    from '/data1/tpch/skew/1/nation.tbl'     ( FORMAT CSV, DELIMITER('|') );
\copy customer  from '/data1/tpch/skew/1/customer.tbl'   ( FORMAT CSV, DELIMITER('|') );
\copy supplier  from '/data1/tpch/skew/1/supplier.tbl'   ( FORMAT CSV, DELIMITER('|') );
\copy part      from '/data1/tpch/skew/1/part.tbl'       ( FORMAT CSV, DELIMITER('|') );
\copy partsupp  from '/data1/tpch/skew/1/partsupp.tbl'   ( FORMAT CSV, DELIMITER('|') );
\copy orders    from '/data1/tpch/skew/1/orders.tbl'     ( FORMAT CSV, DELIMITER('|') );
\copy lineitem  from '/data1/tpch/skew/1/lineitem.tbl'   ( FORMAT CSV, DELIMITER('|') );


