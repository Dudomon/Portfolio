"""
Event Bus implementation using RabbitMQ
Handles publishing and consuming events for microservices communication
"""

import aio_pika
import json
import logging
from typing import Callable, Dict

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self, rabbitmq_url: str):
        self.rabbitmq_url = rabbitmq_url
        self.connection = None
        self.channel = None
        self.exchange = None
        self.consumers: Dict[str, Callable] = {}
    
    async def connect(self):
        """Establish connection to RabbitMQ"""
        try:
            self.connection = await aio_pika.connect_robust(self.rabbitmq_url)
            self.channel = await self.connection.channel()
            
            # Declare exchange for events
            self.exchange = await self.channel.declare_exchange(
                "ecommerce_events",
                aio_pika.ExchangeType.TOPIC,
                durable=True
            )
            
            logger.info("Connected to RabbitMQ")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    async def publish(self, routing_key: str, message: dict):
        """Publish event to exchange"""
        try:
            message_body = json.dumps(message).encode()
            
            await self.exchange.publish(
                aio_pika.Message(
                    body=message_body,
                    content_type="application/json",
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key=routing_key
            )
            
            logger.info(f"Published event: {routing_key}")
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            raise
    
    async def subscribe(self, routing_key: str, callback: Callable):
        """Subscribe to events with routing key pattern"""
        try:
            # Declare queue
            queue = await self.channel.declare_queue(
                f"order_service_{routing_key}",
                durable=True
            )
            
            # Bind queue to exchange
            await queue.bind(self.exchange, routing_key=routing_key)
            
            # Start consuming
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process():
                        try:
                            data = json.loads(message.body.decode())
                            await callback(data)
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
        
        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")
            raise
    
    def is_connected(self) -> bool:
        """Check if connected to RabbitMQ"""
        return self.connection is not None and not self.connection.is_closed
    
    async def close(self):
        """Close connection"""
        if self.connection:
            await self.connection.close()
            logger.info("Disconnected from RabbitMQ")
