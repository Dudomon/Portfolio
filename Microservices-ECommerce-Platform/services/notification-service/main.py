"""
Notification Service
Consumes events and sends notifications (email, SMS, push)
"""

import asyncio
import aio_pika
import json
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotificationService:
    def __init__(self, rabbitmq_url: str):
        self.rabbitmq_url = rabbitmq_url
        self.connection = None
        self.channel = None
    
    async def connect(self):
        """Connect to RabbitMQ"""
        try:
            self.connection = await aio_pika.connect_robust(self.rabbitmq_url)
            self.channel = await self.connection.channel()
            logger.info("Connected to RabbitMQ")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    async def start_consuming(self):
        """Start consuming events"""
        exchange = await self.channel.declare_exchange(
            "ecommerce_events",
            aio_pika.ExchangeType.TOPIC,
            durable=True
        )
        
        # Subscribe to order events
        await self._subscribe(exchange, "order.created", self.handle_order_created)
        await self._subscribe(exchange, "order.completed", self.handle_order_completed)
        await self._subscribe(exchange, "order.cancelled", self.handle_order_cancelled)
        await self._subscribe(exchange, "payment.failed", self.handle_payment_failed)
        
        logger.info("Started consuming events")
    
    async def _subscribe(self, exchange, routing_key: str, callback):
        """Subscribe to specific event"""
        queue = await self.channel.declare_queue(
            f"notification_{routing_key}",
            durable=True
        )
        
        await queue.bind(exchange, routing_key=routing_key)
        
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    try:
                        data = json.loads(message.body.decode())
                        await callback(data)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
    
    async def handle_order_created(self, event_data: Dict):
        """Handle order created event"""
        order_id = event_data.get("order_id")
        user_id = event_data.get("user_id")
        
        logger.info(f"Sending order confirmation for order {order_id} to user {user_id}")
        
        # Simulate sending email
        await self.send_email(
            user_id,
            "Order Confirmation",
            f"Your order {order_id} has been received and is being processed."
        )
    
    async def handle_order_completed(self, event_data: Dict):
        """Handle order completed event"""
        order_id = event_data.get("order_id")
        user_id = event_data.get("user_id")
        
        logger.info(f"Sending order completion notification for order {order_id}")
        
        await self.send_email(
            user_id,
            "Order Completed",
            f"Your order {order_id} has been completed successfully!"
        )
    
    async def handle_order_cancelled(self, event_data: Dict):
        """Handle order cancelled event"""
        order_id = event_data.get("order_id")
        reason = event_data.get("reason", "unknown")
        
        logger.info(f"Sending order cancellation notification for order {order_id}")
        
        await self.send_email(
            "user",
            "Order Cancelled",
            f"Your order {order_id} has been cancelled. Reason: {reason}"
        )
    
    async def handle_payment_failed(self, event_data: Dict):
        """Handle payment failed event"""
        order_id = event_data.get("order_id")
        
        logger.info(f"Sending payment failure notification for order {order_id}")
        
        await self.send_email(
            "user",
            "Payment Failed",
            f"Payment for order {order_id} has failed. Please try again."
        )
    
    async def send_email(self, user_id: str, subject: str, body: str):
        """Simulate sending email"""
        logger.info(f"EMAIL to {user_id}: {subject} - {body}")
        await asyncio.sleep(0.1)  # Simulate email sending delay
    
    async def close(self):
        """Close connection"""
        if self.connection:
            await self.connection.close()


async def main():
    rabbitmq_url = "amqp://guest:guest@rabbitmq:5672/"
    
    service = NotificationService(rabbitmq_url)
    await service.connect()
    
    try:
        await service.start_consuming()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
