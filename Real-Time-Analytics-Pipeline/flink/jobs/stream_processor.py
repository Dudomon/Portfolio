"""
Flink Stream Processing Job
Processes events from Kafka, performs aggregations, and writes to ClickHouse
"""

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaOffsetsInitializer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream.functions import MapFunction, AggregateFunction
from pyflink.datastream.window import TumblingProcessingTimeWindows
from pyflink.common.time import Time
import json
from datetime import datetime


class EventParser(MapFunction):
    """Parse JSON events from Kafka"""
    
    def map(self, value):
        try:
            event = json.loads(value)
            return event
        except Exception as e:
            print(f"Error parsing event: {e}")
            return None


class RevenueAggregator(AggregateFunction):
    """Aggregate revenue and transaction count"""
    
    def create_accumulator(self):
        return {
            'count': 0,
            'revenue': 0.0,
            'users': set()
        }
    
    def add(self, value, accumulator):
        if value and value.get('event_type') == 'purchase':
            accumulator['count'] += 1
            accumulator['revenue'] += float(value.get('revenue', 0))
            accumulator['users'].add(value.get('user_id'))
        return accumulator
    
    def get_result(self, accumulator):
        return {
            'timestamp': datetime.now().isoformat(),
            'total_transactions': accumulator['count'],
            'total_revenue': accumulator['revenue'],
            'unique_users': len(accumulator['users']),
            'avg_order_value': accumulator['revenue'] / accumulator['count'] if accumulator['count'] > 0 else 0
        }
    
    def merge(self, acc1, acc2):
        return {
            'count': acc1['count'] + acc2['count'],
            'revenue': acc1['revenue'] + acc2['revenue'],
            'users': acc1['users'].union(acc2['users'])
        }


class ClickHouseSink(MapFunction):
    """Write aggregated data to ClickHouse"""
    
    def __init__(self):
        self.client = None
    
    def open(self, runtime_context):
        from clickhouse_driver import Client
        self.client = Client(
            host='clickhouse',
            port=9000,
            database='analytics',
            user='admin',
            password='admin123'
        )
    
    def map(self, value):
        try:
            if value:
                query = """
                INSERT INTO metrics_1min 
                (timestamp, event_type, total_events, unique_users, total_revenue, avg_order_value)
                VALUES
                """
                self.client.execute(
                    query,
                    [{
                        'timestamp': value['timestamp'],
                        'event_type': 'purchase',
                        'total_events': value['total_transactions'],
                        'unique_users': value['unique_users'],
                        'total_revenue': value['total_revenue'],
                        'avg_order_value': value['avg_order_value']
                    }]
                )
        except Exception as e:
            print(f"Error writing to ClickHouse: {e}")
        return value


def main():
    # Create execution environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(2)
    
    # Enable checkpointing for fault tolerance
    env.enable_checkpointing(60000)  # checkpoint every 60 seconds
    
    # Configure Kafka source
    kafka_source = KafkaSource.builder() \
        .set_bootstrap_servers('kafka:29092') \
        .set_topics('ecommerce-events') \
        .set_group_id('flink-consumer-group') \
        .set_starting_offsets(KafkaOffsetsInitializer.latest()) \
        .set_value_only_deserializer(SimpleStringSchema()) \
        .build()
    
    # Create data stream
    stream = env.from_source(
        kafka_source,
        watermark_strategy=None,
        source_name='Kafka Source'
    )
    
    # Process stream
    processed_stream = stream \
        .map(EventParser(), output_type=Types.PICKLED_BYTE_ARRAY()) \
        .filter(lambda x: x is not None) \
        .window_all(TumblingProcessingTimeWindows.of(Time.minutes(1))) \
        .aggregate(RevenueAggregator()) \
        .map(ClickHouseSink())
    
    # Execute job
    env.execute('Real-Time Analytics Pipeline')


if __name__ == '__main__':
    main()
