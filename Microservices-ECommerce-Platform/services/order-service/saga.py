"""
Saga Pattern Implementation for Distributed Transactions
Coordinates order processing across multiple services
"""

import logging
from typing import Dict
from event_bus import EventBus
from database import SessionLocal
from models import Order, OrderStatus, OrderEvent

logger = logging.getLogger(__name__)


class OrderSaga:
    """
    Implements Saga pattern for order processing
    
    Flow:
    1. Order Created -> Request Payment
    2. Payment Completed -> Reserve Inventory
    3. Inventory Reserved -> Send Notification
    4. All Steps Complete -> Mark Order as Completed
    
    Compensation (if any step fails):
    - Cancel order
    - Refund payment (if processed)
    - Release inventory (if reserved)
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
    
    async def start_consuming(self):
        """Start consuming events"""
        await self.event_bus.subscribe("payment.completed", self.handle_payment_completed)
        await self.event_bus.subscribe("payment.failed", self.handle_payment_failed)
        await self.event_bus.subscribe("inventory.reserved", self.handle_inventory_reserved)
        await self.event_bus.subscribe("inventory.failed", self.handle_inventory_failed)
    
    async def handle_payment_completed(self, event_data: Dict):
        """Handle successful payment"""
        order_id = event_data.get("order_id")
        logger.info(f"Payment completed for order: {order_id}")
        
        db = SessionLocal()
        try:
            # Update order status
            order = db.query(Order).filter(Order.id == order_id).first()
            if order:
                order.status = OrderStatus.PAYMENT_COMPLETED
                db.commit()
                
                # Store event
                event = OrderEvent(
                    order_id=order_id,
                    event_type="payment_completed",
                    event_data=event_data
                )
                db.add(event)
                db.commit()
                
                # Request inventory reservation
                await self.event_bus.publish("inventory.reserve", {
                    "order_id": order_id,
                    "items": order.items
                })
        finally:
            db.close()
    
    async def handle_payment_failed(self, event_data: Dict):
        """Handle payment failure - compensate"""
        order_id = event_data.get("order_id")
        logger.warning(f"Payment failed for order: {order_id}")
        
        db = SessionLocal()
        try:
            order = db.query(Order).filter(Order.id == order_id).first()
            if order:
                order.status = OrderStatus.PAYMENT_FAILED
                db.commit()
                
                # Store event
                event = OrderEvent(
                    order_id=order_id,
                    event_type="payment_failed",
                    event_data=event_data
                )
                db.add(event)
                db.commit()
                
                # Publish order cancelled event
                await self.event_bus.publish("order.cancelled", {
                    "order_id": order_id,
                    "reason": "payment_failed"
                })
        finally:
            db.close()
    
    async def handle_inventory_reserved(self, event_data: Dict):
        """Handle successful inventory reservation"""
        order_id = event_data.get("order_id")
        logger.info(f"Inventory reserved for order: {order_id}")
        
        db = SessionLocal()
        try:
            order = db.query(Order).filter(Order.id == order_id).first()
            if order:
                order.status = OrderStatus.COMPLETED
                db.commit()
                
                # Store event
                event = OrderEvent(
                    order_id=order_id,
                    event_type="inventory_reserved",
                    event_data=event_data
                )
                db.add(event)
                db.commit()
                
                # Publish order completed event
                await self.event_bus.publish("order.completed", {
                    "order_id": order_id,
                    "user_id": order.user_id,
                    "total_amount": float(order.total_amount)
                })
        finally:
            db.close()
    
    async def handle_inventory_failed(self, event_data: Dict):
        """Handle inventory reservation failure - compensate"""
        order_id = event_data.get("order_id")
        logger.warning(f"Inventory reservation failed for order: {order_id}")
        
        db = SessionLocal()
        try:
            order = db.query(Order).filter(Order.id == order_id).first()
            if order:
                order.status = OrderStatus.FAILED
                db.commit()
                
                # Store event
                event = OrderEvent(
                    order_id=order_id,
                    event_type="inventory_failed",
                    event_data=event_data
                )
                db.add(event)
                db.commit()
                
                # Compensate: refund payment
                await self.event_bus.publish("payment.refund", {
                    "order_id": order_id,
                    "amount": float(order.total_amount)
                })
                
                # Publish order cancelled event
                await self.event_bus.publish("order.cancelled", {
                    "order_id": order_id,
                    "reason": "inventory_unavailable"
                })
        finally:
            db.close()
