from joins.table import Table


def train_stats():
    table = Table()
    table.fit('data/pm25_100.csv', join_keys=['PRES'])
