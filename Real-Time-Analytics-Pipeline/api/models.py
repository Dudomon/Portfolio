"""Pydantic models for API"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from decimal import Decimal


class Event(BaseModel):
    event_id: str
    event_type: str = Field(..., description="Type of event: purchase, view, cart_add, etc.")
    timestamp: datetime = Field(default_factory=datetime.now)
    user_id: str
    session_id: str
    product_id: Optional[str] = None
    product_name: Optional[str] = None
    category: Optional[str] = None
    price: Optional[Decimal] = None
    quantity: Optional[int] = 1
    revenue: Optional[Decimal] = None
    country: Optional[str] = None
    city: Optional[str] = None
    device_type: Optional[str] = None
    browser: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}


class MetricQuery(BaseModel):
    metric_name: str
    start_time: datetime
    end_time: datetime
    aggregation: str = "sum"  # sum, avg, count, min, max
    group_by: Optional[str] = None


class AlertConfig(BaseModel):
    alert_type: str
    metric_name: str
    threshold: float
    operator: str = "greater_than"  # greater_than, less_than, equals
    window_seconds: int = 60
    enabled: bool = True
