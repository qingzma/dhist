from joins.table import TableContainer


def train_stats():
    table = TableContainer()
    table.fit('data/pm25_100.csv', join_keys=['PRES'])
