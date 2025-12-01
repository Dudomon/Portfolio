"""ClickHouse database client"""

from clickhouse_driver import Client
from config import settings


class ClickHouseClient:
    def __init__(self):
        self.client = Client(
            host=settings.CLICKHOUSE_HOST,
            port=9000,  # Native protocol port
            database=settings.CLICKHOUSE_DB,
            user=settings.CLICKHOUSE_USER,
            password=settings.CLICKHOUSE_PASSWORD
        )
    
    def execute(self, query, params=None):
        """Execute a query"""
        return self.client.execute(query, params)
    
    def execute_iter(self, query, params=None):
        """Execute query and return iterator"""
        return self.client.execute_iter(query, params)
    
    def insert(self, table, data):
        """Insert data into table"""
        self.client.execute(f"INSERT INTO {table} VALUES", data)
