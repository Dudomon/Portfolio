"""
Order Service - Main Application
Handles order creation, management, and event sourcing
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional
import asyncio
import logging

from config import settings
from database import get_db, init_db
from models import Order, OrderCreate, OrderStatus, OrderEvent
from event_bus import EventBus
from saga import OrderSaga

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

event_bus = None
order_saga = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global event_bus, order_saga
    
    # Startup
    logger.info("Starting Order Service...")
    await init_db()
    
    event_bus = EventBus(settings.RABBITMQ_URL)
    await event_bus.connect()
    
    order_saga = OrderSaga(event_bus)
    asyncio.create_task(order_saga.start_consuming())
    
    logger.info("Order Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Order Service...")
    await event_bus.close()


app = FastAPI(
    title="Order Service",
    description="Microservice for order management with event sourcing",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "order-service",
        "version": "1.0.0"
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        # Check database connection
        db = next(get_db())
        db.execute("SELECT 1")
        
        # Check event bus connection
        if not event_bus or not event_bus.is_connected():
            raise Exception("Event bus not connected")
        
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@app.post("/orders", response_model=Order, status_code=201)
async def create_order(order_data: OrderCreate, db=Depends(get_db)):
    """
    Create a new order and initiate saga
    
    This endpoint creates an order and publishes an OrderCreated event
    which triggers the distributed transaction saga pattern.
    """
    try:
        # Create order in database
        order = Order(
            user_id=order_data.user_id,
            items=order_data.items,
            total_amount=sum(item.price * item.quantity for item in order_data.items),
            status=OrderStatus.PENDING
        )
        
        db.add(order)
        db.commit()
        db.refresh(order)
        
        # Publish OrderCreated event
        await event_bus.publish("order.created", {
            "order_id": str(order.id),
            "user_id": order.user_id,
            "items": [item.dict() for item in order.items],
            "total_amount": float(order.total_amount),
            "timestamp": order.created_at.isoformat()
        })
        
        logger.info(f"Order created: {order.id}")
        return order
        
    except Exception as e:
        logger.error(f"Error creating order: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create order")


@app.get("/orders/{order_id}", response_model=Order)
async def get_order(order_id: str, db=Depends(get_db)):
    """Get order by ID"""
    order = db.query(Order).filter(Order.id == order_id).first()
    
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    return order


@app.get("/orders", response_model=List[Order])
async def list_orders(
    user_id: Optional[str] = None,
    status: Optional[OrderStatus] = None,
    skip: int = 0,
    limit: int = 100,
    db=Depends(get_db)
):
    """List orders with optional filters"""
    query = db.query(Order)
    
    if user_id:
        query = query.filter(Order.user_id == user_id)
    
    if status:
        query = query.filter(Order.status == status)
    
    orders = query.offset(skip).limit(limit).all()
    return orders


@app.patch("/orders/{order_id}/status")
async def update_order_status(
    order_id: str,
    status: OrderStatus,
    db=Depends(get_db)
):
    """Update order status"""
    order = db.query(Order).filter(Order.id == order_id).first()
    
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    old_status = order.status
    order.status = status
    db.commit()
    
    # Publish status change event
    await event_bus.publish("order.status_changed", {
        "order_id": str(order.id),
        "old_status": old_status.value,
        "new_status": status.value,
        "timestamp": order.updated_at.isoformat()
    })
    
    logger.info(f"Order {order_id} status changed: {old_status} -> {status}")
    return {"status": "updated"}


@app.get("/orders/{order_id}/events", response_model=List[OrderEvent])
async def get_order_events(order_id: str, db=Depends(get_db)):
    """
    Get event history for an order (Event Sourcing)
    
    Returns all events that have occurred for this order,
    allowing reconstruction of order state at any point in time.
    """
    events = db.query(OrderEvent).filter(
        OrderEvent.order_id == order_id
    ).order_by(OrderEvent.created_at).all()
    
    return events


@app.get("/metrics")
async def get_metrics(db=Depends(get_db)):
    """Get service metrics"""
    total_orders = db.query(Order).count()
    pending_orders = db.query(Order).filter(Order.status == OrderStatus.PENDING).count()
    completed_orders = db.query(Order).filter(Order.status == OrderStatus.COMPLETED).count()
    failed_orders = db.query(Order).filter(Order.status == OrderStatus.FAILED).count()
    
    return {
        "total_orders": total_orders,
        "pending_orders": pending_orders,
        "completed_orders": completed_orders,
        "failed_orders": failed_orders
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
