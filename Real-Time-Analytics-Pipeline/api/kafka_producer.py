"""Kafka event producer"""

from kafka import KafkaProducer
import json
from config import settings


class EventProducer:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3
        )
    
    def send_event(self, event):
        """Send event to Kafka topic"""
        future = self.producer.send(settings.KAFKA_TOPIC, value=event)
        return future.get(timeout=10)
    
    def close(self):
        """Close producer"""
        self.producer.close()
