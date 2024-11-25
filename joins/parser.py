import csv
import os
import time

import numpy as np
import sqlparse
from sqlparse.tokens import Token
import copy

from joins.base_logger import logger
from joins.join_graph import process_condition
from joins.schema_base import (
    AggregationOperationType,
    AggregationType,
    Query,
    QueryType,
)


def timestamp_transform(time_string, start_date="2010-07-19 00:00:00"):
    start_date_int = time.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    time_array = time.strptime(time_string, "'%Y-%m-%d %H:%M:%S'")
    return int(time.mktime(time_array)) - int(time.mktime(start_date_int))


def _extract_identifiers(tokens, enforce_single=True):
    identifiers = [
        token for token in tokens if isinstance(token, sqlparse.sql.IdentifierList)
    ]
    if len(identifiers) >= 1:
        if enforce_single:
            assert len(identifiers) == 1
        identifiers = identifiers[0]
    else:
        identifiers = [
            token for token in tokens if isinstance(token, sqlparse.sql.Identifier)
        ]
    return identifiers


# Find corresponding table of attribute
def _find_matching_table(attribute, schema, alias_dict):
    table_name = None
    for table_obj in schema.tables:
        if table_obj.table_name not in alias_dict.keys():
            continue
        if attribute in table_obj.attributes:
            table_name = table_obj.table_name

    assert table_name is not None, f"No table found for attribute {attribute}."
    return table_name


def _fully_qualified_attribute_name(identifier, schema, alias_dict, return_split=False):
    if len(identifier.tokens) == 1:
        attribute = identifier.tokens[0].value
        table_name = _find_matching_table(attribute, schema, alias_dict)
        if not return_split:
            return table_name + "." + attribute
        else:
            return table_name, attribute

    # Replace alias by full table names
    assert identifier.tokens[1].value == ".", "Invalid Identifier"
    if not return_split:
        return alias_dict[identifier.tokens[0].value] + "." + identifier.tokens[2].value
    else:
        return alias_dict[identifier.tokens[0].value], identifier.tokens[2].value


def _parse_aggregation(alias_dict, function, query, schema):
    operation_factors = []
    operation_type = None
    operator = _extract_identifiers(function.tokens)[0]
    if operator.normalized == "sum" or operator.normalized == "SUM":
        operation_type = AggregationType.SUM
    elif operator.normalized == "avg" or operator.normalized == "AVG":
        operation_type = AggregationType.AVG
    elif operator.normalized == "count" or operator.normalized == "COUNT":
        query.add_aggregation_operation(
            (AggregationOperationType.AGGREGATION, AggregationType.COUNT, [])
        )
        return
    else:
        raise Exception(f"Unknown operator: {operator.normalized} ")
    operand_parantheses = [
        token for token in function if isinstance(token, sqlparse.sql.Parenthesis)
    ]
    assert len(operand_parantheses) == 1
    operand_parantheses = operand_parantheses[0]
    operation_tokens = [
        token
        for token in operand_parantheses
        if isinstance(token, sqlparse.sql.Operation)
    ]
    # Product of columns
    if len(operation_tokens) == 1:
        operation_tokens = operation_tokens[0].tokens
        assert [
            token.value == " " or token.value == "*"
            for token in operation_tokens
            if not isinstance(token, sqlparse.sql.Identifier)
        ], "Currently multiplication is the only supported operator."
        identifiers = _extract_identifiers(operation_tokens)
        for identifier in identifiers:
            feature = _fully_qualified_attribute_name(
                identifier, schema, alias_dict, return_split=True
            )
            operation_factors.append(feature)
    # single column
    else:
        feature = _fully_qualified_attribute_name(
            _extract_identifiers(operand_parantheses)[0],
            schema,
            alias_dict,
            return_split=True,
        )
        operation_factors.append(feature)
    query.add_aggregation_operation(
        (AggregationOperationType.AGGREGATION, operation_type, operation_factors)
    )


def parse_what_if_query(query_str, schema, return_condition_string=False):
    assert query_str.startswith("WHAT IF"), "Not a valid what if query"
    query_str = query_str.replace("WHAT IF", "")

    # find out factor
    if "DECREASE BY" in query_str:
        percentage_change = -1
        condition_string, percentage = query_str.split("DECREASE BY")
    elif "INCREASE BY" in query_str:
        percentage_change = 1
        condition_string, percentage = query_str.split("INCREASE BY")
    else:
        raise ValueError("Not a valid what if query")
    percentage = float(percentage.strip(" %;")) / 100
    percentage_change *= percentage

    # parse condtions
    parsed_conditions = []
    conditions = condition_string.split(" AND ")
    for condition in conditions:
        if "=" in condition:
            operator = "="
            column, where_condition = condition.split("=", 1)
        elif "IN" in condition:
            operator = "IN"
            column, where_condition = condition.split("IN", 1)
        else:
            raise NotImplementedError

        column = column.strip()
        where_condition = where_condition.strip()

        if "." in column:
            table, attribute = column.split(".", 1)
        else:
            table = _find_matching_table(
                column,
                schema,
                {table.table_name: table.table_name for table in schema.tables},
            )
            attribute = column

        parsed_conditions.append(
            (table, attribute + " " + operator + " " + where_condition)
        )

    if return_condition_string:
        return parsed_conditions, percentage_change, condition_string
    return parsed_conditions, percentage_change


def all_operations_of_type(type, query):
    return all(
        [
            aggregation_type == type
            for aggregation_operation_type, aggregation_type, _ in query.aggregation_operations
            if aggregation_operation_type == AggregationOperationType.AGGREGATION
        ]
    )


def parse_query(query_str, schema):
    """
    Parses simple SQL queries and returns cardinality query object.
    :param query_str:
    :param schema:
    :return:
    """
    query = Query(schema)

    # split query into part before from
    parsed_tokens = sqlparse.parse(query_str)[0]
    from_idxs = [
        i for i, token in enumerate(parsed_tokens) if token.normalized == "FROM"
    ]
    assert len(from_idxs) == 1, "Nested queries are currently not supported."
    from_idx = from_idxs[0]
    tokens_before_from = parsed_tokens[:from_idx]

    # split query into part after from and before group by
    group_by_idxs = [
        i for i, token in enumerate(parsed_tokens) if token.normalized == "GROUP BY"
    ]
    assert (
        len(group_by_idxs) == 0 or len(group_by_idxs) == 1
    ), "Nested queries are currently not supported."
    group_by_attributes = None
    if len(group_by_idxs) == 1:
        tokens_from_from = parsed_tokens[from_idx: group_by_idxs[0]]
        order_by_idxs = [
            i for i, token in enumerate(parsed_tokens) if token.normalized == "ORDER BY"
        ]
        if len(order_by_idxs) > 0:
            group_by_end = order_by_idxs[0]
            tokens_group_by = parsed_tokens[group_by_idxs[0]: group_by_end]
        else:
            tokens_group_by = parsed_tokens[group_by_idxs[0]:]
        # Do not enforce single because there could be order by statement. Will be ignored.
        group_by_attributes = _extract_identifiers(
            tokens_group_by, enforce_single=False
        )
    else:
        tokens_from_from = parsed_tokens[from_idx:]

    # Get identifier to obtain relevant tables
    identifiers = _extract_identifiers(tokens_from_from)
    identifier_token_length = [
        len(token.tokens)
        for token in identifiers
        if isinstance(token, sqlparse.sql.Identifier)
    ][0]

    if identifier_token_length == 3:
        # (title t)
        tables = [
            (token[0].value, token[2].value)
            for token in identifiers
            if isinstance(token, sqlparse.sql.Identifier)
        ]
    elif identifier_token_length == 5:
        # (title as t)
        tables = [
            (token[0].value, token[-1].value)
            for token in identifiers
            if isinstance(token, sqlparse.sql.Identifier)
        ]
    else:
        # (title, title), no alias
        tables = [
            (token[0].value, token[0].value)
            for token in identifiers
            if isinstance(token, sqlparse.sql.Identifier)
        ]
    alias_dict = dict()
    for table, alias in tables:
        query.table_set.add(table)
        alias_dict[alias] = table

    # If there is a group by clause, parse it
    if group_by_attributes is not None:
        identifier_token_length = [
            len(token.tokens) for token in _extract_identifiers(group_by_attributes)
        ][0]

        if identifier_token_length == 3:
            # lo.d_year
            group_by_attributes = [
                (alias_dict[token[0].value], token[2].value)
                for token in _extract_identifiers(group_by_attributes)
            ]
            for table, attribute in group_by_attributes:
                query.add_group_by(table, attribute)
        else:
            # d_year
            for group_by_token in _extract_identifiers(group_by_attributes):
                attribute = group_by_token.value
                table = _find_matching_table(attribute, schema, alias_dict)
                query.add_group_by(table, attribute)

    # Obtain projection/ aggregation attributes
    count_statements = [
        token
        for token in tokens_before_from
        if token.normalized == "COUNT(*)" or token.normalized == "count(*)"
    ]
    assert (
        len(count_statements) <= 1
    ), "Several count statements are currently not supported."
    if len(count_statements) == 1:
        query.query_type = QueryType.CARDINALITY
    else:
        query.query_type = QueryType.AQP
        identifiers = _extract_identifiers(tokens_before_from)

        # Only aggregation attribute, e.g. sum(lo_extendedprice*lo_discount)
        if not isinstance(identifiers, sqlparse.sql.IdentifierList):
            handle_aggregation(alias_dict, query, schema, tokens_before_from)
        # group by attributes and aggregation attribute
        else:
            handle_aggregation(alias_dict, query, schema, identifiers.tokens)

    # Obtain where statements
    where_statements = [
        token for token in tokens_from_from if isinstance(token, sqlparse.sql.Where)
    ]
    assert len(where_statements) <= 1
    if len(where_statements) == 0:
        return query

    where_statements = where_statements[0]
    assert (
        len([token for token in where_statements if token.normalized == "OR"]) == 0
    ), "OR statements currently unsupported."

    # Parse where statements
    # parse multiple values differently because sqlparse does not parse as comparison
    in_statements = [
        idx for idx, token in enumerate(where_statements) if token.normalized == "IN"
    ]
    for in_idx in in_statements:
        assert where_statements.tokens[in_idx - 1].value == " "
        assert where_statements.tokens[in_idx + 1].value == " "
        # ('bananas', 'apples')
        possible_values = where_statements.tokens[in_idx + 2]
        assert isinstance(possible_values, sqlparse.sql.Parenthesis)
        # fruits
        identifier = where_statements.tokens[in_idx - 2]
        assert isinstance(identifier, sqlparse.sql.Identifier)

        if len(identifier.tokens) == 1:
            left_table_name, left_attribute = _fully_qualified_attribute_name(
                identifier, schema, alias_dict, return_split=True
            )
            query.add_where_condition(
                left_table_name, left_attribute + " IN " + possible_values.value
            )

        else:
            assert identifier.tokens[1].value == ".", "Invalid identifier."
            # Replace alias by full table names
            query.add_where_condition(
                alias_dict[identifier.tokens[0].value],
                identifier.tokens[2].value + " IN " + possible_values.value,
            )
    # normal comparisons
    comparisons = [
        token
        for token in where_statements
        if isinstance(token, sqlparse.sql.Comparison)
    ]
    for comparison in comparisons:
        left = comparison.left
        assert isinstance(
            left, sqlparse.sql.Identifier), "Invalid where condition"
        comparison_tokens = [
            token
            for token in comparison.tokens
            if token.ttype == Token.Operator.Comparison
        ]
        assert len(comparison_tokens) == 1, "Invalid comparison"
        operator_idx = comparison.tokens.index(comparison_tokens[0])

        if len(left.tokens) == 1:
            left_table_name, left_attribute = _fully_qualified_attribute_name(
                left, schema, alias_dict, return_split=True
            )
            left_part = left_table_name + "." + left_attribute
            right = comparison.right

            # Join relationship
            if isinstance(right, sqlparse.sql.Identifier):
                assert len(right.tokens) == 1, "Invalid Identifier"
                right_attribute = right.tokens[0].value
                right_table_name = _find_matching_table(
                    right_attribute, schema, alias_dict
                )
                right_part = right_table_name + "." + right_attribute

                assert (
                    comparison.tokens[operator_idx].value == "="
                ), "Invalid join condition"
                assert (
                    left_part + " = " + right_part
                    in schema.relationship_dictionary.keys()
                    or right_part + " = " + left_part
                    in schema.relationship_dictionary.keys()
                ), "Relationship unknown"
                if (
                    left_part + " = " + right_part
                    in schema.relationship_dictionary.keys()
                ):
                    query.add_join_condition(left_part + " = " + right_part)
                elif (
                    right_part + " = " + left_part
                    in schema.relationship_dictionary.keys()
                ):
                    query.add_join_condition(right_part + " = " + left_part)

            # Where condition
            else:
                where_condition = left_attribute + "".join(
                    [token.value.strip()
                     for token in comparison.tokens[operator_idx:]]
                )
                query.add_where_condition(left_table_name, where_condition)

        else:
            # Replace alias by full table names
            left_part = _fully_qualified_attribute_name(
                left, schema, alias_dict)

            right = comparison.right
            # Join relationship
            if isinstance(right, sqlparse.sql.Identifier):
                if right.tokens[1].value == ".":
                    right_part = (
                        alias_dict[right.tokens[0].value] +
                        "." + right.tokens[2].value
                    )
                    assert (
                        comparison.tokens[operator_idx].value == "="
                    ), "Invalid join condition"
                    assert (
                        left_part + " = " + right_part
                        in schema.relationship_dictionary.keys()
                        or right_part + " = " + left_part
                        in schema.relationship_dictionary.keys()
                    ), f"Relationship unknown{right_part + ' = ' + left_part}"
                    if (
                        left_part + " = " + right_part
                        in schema.relationship_dictionary.keys()
                    ):
                        query.add_join_condition(
                            left_part + " = " + right_part)
                    elif (
                        right_part + " = " + left_part
                        in schema.relationship_dictionary.keys()
                    ):
                        query.add_join_condition(
                            right_part + " = " + left_part)
                else:
                    assert right.tokens[1].value == "::"
                    right.value = str(
                        timestamp_transform(right.tokens[0].value))
                    query.add_where_condition(
                        alias_dict[left.tokens[0].value],
                        left.tokens[2].value
                        + comparison.tokens[operator_idx].value
                        + right.value,
                    )
            # Where condition
            else:
                query.add_where_condition(
                    alias_dict[left.tokens[0].value],
                    left.tokens[2].value
                    + comparison.tokens[operator_idx].value
                    + right.value,
                )

    return query


def handle_aggregation(alias_dict, query, schema, tokens_before_from):
    operations = [
        token
        for token in tokens_before_from
        if isinstance(token, sqlparse.sql.Operation)
    ]
    assert len(operations) <= 1, "A maximum of 1 operation is supported."
    if len(operations) == 0:
        functions = [
            token
            for token in tokens_before_from
            if isinstance(token, sqlparse.sql.Function)
        ]
        assert len(
            functions) == 1, "Only a single aggregate function is supported."
        function = functions[0]
        _parse_aggregation(alias_dict, function, query, schema)
    else:
        operation = operations[0]
        inner_operations = [
            token
            for token in operation.tokens
            if isinstance(token, sqlparse.sql.Operation)
        ]
        # handle inner operations recursively
        if len(inner_operations) > 0:
            assert len(
                inner_operations) == 1, "Multiple inner operations impossible"
            handle_aggregation(alias_dict, query, schema, inner_operations)
        for token in operation.tokens:
            if isinstance(token, sqlparse.sql.Function):
                _parse_aggregation(alias_dict, token, query, schema)
            elif token.value == "-":
                query.add_aggregation_operation(
                    (AggregationOperationType.MINUS, None, None)
                )
            elif token.value == "+":
                query.add_aggregation_operation(
                    (AggregationOperationType.PLUS, None, None)
                )


def save_csv(csv_rows, target_csv_path):
    os.makedirs(os.path.dirname(target_csv_path), exist_ok=True)
    logger.info(f"Saving results to {target_csv_path}")

    with open(target_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, csv_rows[0].keys())
        for i, row in enumerate(csv_rows):
            if i == 0:
                w.writeheader()
            w.writerow(row)


def parse_query_simple(query):
    """
    If your selection query contains no aggregation and nested sub-queries, you can use this function to parse a
    join query. Otherwise, use parse_query function.
    """
    # if self.bns is None:
    #     # this is a hack, since we currently only implemented the bayescard and sampling for single table estimate
    #     tables_all, join_cond, join_keys = parse_query_all_join(query)
    #     table_filters = dict()
    #     return tables_all, table_filters, join_cond, join_keys

    query = query.replace(" where ", " WHERE ")
    query = query.replace(" from ", " FROM ")
    query = query.replace(" and ", " AND ")
    query = query.split(";")[0]
    query = query.strip()
    tables_all = {}
    join_cond = []
    table_query = {}
    join_keys = {}
    tables_str = query.split(" WHERE ")[0].split(" FROM ")[-1]
    for table_str in tables_str.split(","):
        table_str = table_str.strip()
        if " as " in table_str:
            tables_all[table_str.split(
                " as ")[-1]] = table_str.split(" as ")[0]
        else:
            tables_all[table_str.split(" ")[-1]] = table_str.split(" ")[0]

    # processing conditions
    if " WHERE " not in query:
        return tables_all, None, None, None
    conditions = query.split(" WHERE ")[-1].split(" AND ")
    for cond in conditions:
        table, cond, join, join_key = process_condition(cond, tables_all)
        if not join:
            attr = cond[0]
            op = cond[1]
            value = cond[2]
            if "Date" in attr:
                assert (
                    "::timestamp" in value
                )  # this is hardcoded for STATS-CEB workload
                value = timestamp_transform(
                    value.strip().split("::timestamp")[0])
            if table not in table_query:
                table_query[table] = dict()
            # construct_table_query(
            #     self.bns[table], table_query[table], attr, op, value)
            if attr not in table_query[table]:
                table_query[table][attr] = dict()
            table_query[table][attr][op] = value
        else:
            join_cond.append(cond)
            for tab in join_key:
                if tab in join_keys:
                    join_keys[tab].add(join_key[tab])
                else:
                    join_keys[tab] = set([join_key[tab]])

    return tables_all, table_query, join_cond, join_keys


def construct_table_query(BN, table_query, attr, ops, val, epsilon=1e-6):
    # if BN is None or attr not in BN.attr_type:
    #     return None
    if BN.attr_type[attr] == "continuous":
        if ops == ">=":
            query_domain = (val, np.infty)
        elif ops == ">":
            query_domain = (val + epsilon, np.infty)
        elif ops == "<=":
            query_domain = (-np.infty, val)
        elif ops == "<":
            query_domain = (-np.infty, val - epsilon)
        elif ops == "=" or ops == "==":
            query_domain = val
        else:
            assert False, f"operation {ops} is invalid for continous domain"

        if attr not in table_query:
            table_query[attr] = query_domain
        else:
            prev_l = table_query[attr][0]
            prev_r = table_query[attr][1]
            query_domain = (max(prev_l, query_domain[0]), min(
                prev_r, query_domain[1]))
            table_query[attr] = query_domain

    else:
        attr_domain = BN.domain[attr]
        if type(attr_domain[0]) != str:
            attr_domain = np.asarray(attr_domain)
        if ops == "in":
            assert type(val) == list, "use list for in query"
            query_domain = val
        elif ops == "=" or ops == "==":
            if type(val) == list:
                query_domain = val
            else:
                query_domain = [val]
        else:
            if type(val) == list:
                assert len(val) == 1
                val = val[0]
                assert type(val) == int or type(val) == float
            operater = OPS[ops]
            query_domain = list(attr_domain[operater(attr_domain, val)])

        if attr not in table_query:
            table_query[attr] = query_domain
        else:
            query_domain = [i for i in query_domain if i in table_query[attr]]
            table_query[attr] = query_domain

    return table_query


OPS = {
    ">": np.greater,
    "<": np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "=": np.equal,
    "==": np.equal,
}


def parse_single_table_query(query):
    tables_all, table_query, _, _ = parse_query_simple(query)
    tbl_name = list(tables_all.values())[0]

    if not table_query:
        return tbl_name, None

    cols = [c.split(".")[1] for c in tuple(table_query[tbl_name].keys())]
    cols.sort()

    return tbl_name, tuple(cols)


def get_two_chained_query(join_keys, join_cond, include_self=True):
    """split the query and get the two invididual join paths that involves only 1 common join key

    Args:
        join_keys (_type_): _description_
        join_cond (_type_): _description_
    Returns:
        Two individual join paths.
    """
    target_tbl = None
    target_jk = None
    for tbl in join_keys:
        jk = join_keys[tbl]
        if len(jk) == 2:
            target_tbl = tbl
            target_jk = jk
            break
    assert target_tbl is not None
    target_jk = list(
        map(lambda kv: kv.replace(".", "").replace(target_tbl, ""), target_jk)
    )

    join_conds = copy.deepcopy(join_cond)
    join_paths = {}
    join_path_alias_items = {}
    for jk in target_jk:
        join_paths[jk] = []
        join_path_alias_items[jk] = []
    # print("target_tbl is ", target_tbl)
    # print("target_jk is ", target_jk)

    while join_conds:
        # print("!!!!!!!!!!!!!")
        # print("join_conds", join_conds)
        idxs_to_remove = []
        for i, cond in enumerate(join_conds):
            # print("hh", join_conds)
            for jk in target_jk:
                # print("jk", jk)
                target_jk_full_item = target_tbl + "." + jk
                copies = copy.deepcopy(join_path_alias_items[jk])
                for alias in join_path_alias_items[jk]:
                    if alias in cond:
                        # print("hehre")
                        # print("alias in cond", alias, cond)
                        another = cond.replace(alias, "").replace(
                            "=", ""
                        )
                        another = another.strip()  # target_jk_full_item
                        tbl_i = another.split(".")[0]
                        join_paths[jk].append(tbl_i)
                        copies.append(another)
                        idxs_to_remove.append(i)
                        # print("join_path_alias_items", join_path_alias_items)
                        # print("idxs_to_remove", idxs_to_remove)
                join_path_alias_items[jk] = copies

                if target_jk_full_item in cond:
                    another = cond.replace(
                        target_jk_full_item, "").replace("=", "")
                    another = another.strip()
                    tbl_i = another.split(".")[0]
                    join_paths[jk].append(tbl_i)
                    join_path_alias_items[jk].append(another)
                    idxs_to_remove.append(i)
        for index in sorted(idxs_to_remove, reverse=True):
            del join_conds[index]
        # print("changed to ", join_conds)

    # add self table if needed
    if include_self:
        for jk in target_jk:
            join_paths[jk].append(target_tbl)
            # tmp = join_paths[jk]
            # tmp.append(target_tbl)
            # join_paths[jk] = tmp
    # print("join_paths", join_paths)
    # print("join_path_alias_items", join_path_alias_items)
    # print("idxs_to_remove", idxs_to_remove)
    return target_tbl, join_paths


class Node:
    def __init__(self, table, join_key):
        self.table = table
        self.join_key = [join_key]
        self.join_paths = {}

    def __repr__(self) -> str:
        return f"[{self.table}-()]"


def get_max_dim(join_keys):
    if not join_keys:
        return 1
    n = list(map(lambda x: len(x), join_keys.values()))
    # print("n is !!!!", n)
    if n:
        return max(n)
    return 1
