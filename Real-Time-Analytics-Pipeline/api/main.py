"""
FastAPI Backend for Real-Time Analytics Pipeline
Provides REST API and WebSocket endpoints for dashboard
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import asyncio
import json
from datetime import datetime, timedelta

from config import settings
from database import ClickHouseClient
from kafka_producer import EventProducer
from redis_client import RedisClient
from models import Event, MetricQuery, AlertConfig


# Connection managers
clickhouse_client = None
kafka_producer = None
redis_client = None
websocket_connections: List[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global clickhouse_client, kafka_producer, redis_client
    
    # Startup
    clickhouse_client = ClickHouseClient()
    kafka_producer = EventProducer()
    redis_client = RedisClient()
    
    # Start background task for real-time updates
    asyncio.create_task(broadcast_metrics())
    
    yield
    
    # Shutdown
    kafka_producer.close()


app = FastAPI(
    title="Real-Time Analytics API",
    description="Backend API for streaming analytics dashboard",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REST API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Real-Time Analytics API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    health_status = {
        "api": "healthy",
        "clickhouse": "unknown",
        "kafka": "unknown",
        "redis": "unknown"
    }
    
    try:
        clickhouse_client.execute("SELECT 1")
        health_status["clickhouse"] = "healthy"
    except:
        health_status["clickhouse"] = "unhealthy"
    
    try:
        redis_client.ping()
        health_status["redis"] = "healthy"
    except:
        health_status["redis"] = "unhealthy"
    
    return health_status


@app.post("/events")
async def ingest_event(event: Event):
    """Ingest a single event into Kafka"""
    try:
        kafka_producer.send_event(event.dict())
        return {"status": "success", "event_id": event.event_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/events/batch")
async def ingest_events_batch(events: List[Event]):
    """Ingest multiple events into Kafka"""
    try:
        for event in events:
            kafka_producer.send_event(event.dict())
        return {"status": "success", "count": len(events)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/realtime")
async def get_realtime_metrics():
    """Get real-time metrics from cache"""
    try:
        cached = redis_client.get("metrics:realtime")
        if cached:
            return json.loads(cached)
        
        # Fallback to ClickHouse
        query = """
        SELECT
            count() as total_events,
            uniq(user_id) as unique_users,
            sum(revenue) as total_revenue,
            avg(price) as avg_order_value
        FROM events
        WHERE timestamp >= now() - INTERVAL 1 MINUTE
        """
        result = clickhouse_client.execute(query)
        
        if result:
            metrics = {
                "total_events": result[0][0],
                "unique_users": result[0][1],
                "total_revenue": float(result[0][2] or 0),
                "avg_order_value": float(result[0][3] or 0),
                "timestamp": datetime.now().isoformat()
            }
            redis_client.setex("metrics:realtime", 5, json.dumps(metrics))
            return metrics
        
        return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/timeseries")
async def get_timeseries_metrics(
    window: str = "1h",
    metric: str = "revenue"
):
    """Get time-series metrics"""
    try:
        # Map window to interval
        interval_map = {
            "1m": "1 MINUTE",
            "5m": "5 MINUTE",
            "15m": "15 MINUTE",
            "1h": "1 HOUR",
            "24h": "1 DAY"
        }
        
        interval = interval_map.get(window, "1 HOUR")
        
        query = f"""
        SELECT
            toStartOfInterval(timestamp, INTERVAL {interval}) as time_bucket,
            count() as events,
            sum(revenue) as revenue,
            uniq(user_id) as users
        FROM events
        WHERE timestamp >= now() - INTERVAL 24 HOUR
        GROUP BY time_bucket
        ORDER BY time_bucket
        """
        
        results = clickhouse_client.execute(query)
        
        data = [
            {
                "timestamp": row[0].isoformat(),
                "events": row[1],
                "revenue": float(row[2] or 0),
                "users": row[3]
            }
            for row in results
        ]
        
        return {"data": data, "window": window}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/top")
async def get_top_products(limit: int = 10):
    """Get top selling products"""
    try:
        query = f"""
        SELECT
            product_id,
            any(product_name) as name,
            any(category) as category,
            count() as sales_count,
            sum(revenue) as total_revenue
        FROM events
        WHERE event_type = 'purchase'
            AND timestamp >= now() - INTERVAL 1 HOUR
        GROUP BY product_id
        ORDER BY sales_count DESC
        LIMIT {limit}
        """
        
        results = clickhouse_client.execute(query)
        
        products = [
            {
                "product_id": row[0],
                "name": row[1],
                "category": row[2],
                "sales_count": row[3],
                "total_revenue": float(row[4] or 0)
            }
            for row in results
        ]
        
        return {"products": products}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/geo/distribution")
async def get_geo_distribution():
    """Get geographic distribution of sales"""
    try:
        query = """
        SELECT
            country,
            count() as events,
            sum(revenue) as revenue,
            uniq(user_id) as users
        FROM events
        WHERE timestamp >= now() - INTERVAL 1 HOUR
        GROUP BY country
        ORDER BY revenue DESC
        """
        
        results = clickhouse_client.execute(query)
        
        distribution = [
            {
                "country": row[0],
                "events": row[1],
                "revenue": float(row[2] or 0),
                "users": row[3]
            }
            for row in results
        ]
        
        return {"distribution": distribution}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts")
async def get_alerts(limit: int = 50):
    """Get recent alerts"""
    try:
        query = f"""
        SELECT
            alert_id,
            alert_type,
            severity,
            timestamp,
            metric_name,
            metric_value,
            threshold,
            message,
            resolved
        FROM alerts
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
        
        results = clickhouse_client.execute(query)
        
        alerts = [
            {
                "alert_id": row[0],
                "alert_type": row[1],
                "severity": row[2],
                "timestamp": row[3].isoformat(),
                "metric_name": row[4],
                "metric_value": float(row[5]),
                "threshold": float(row[6]),
                "message": row[7],
                "resolved": row[8]
            }
            for row in results
        ]
        
        return {"alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time updates"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)


async def broadcast_metrics():
    """Background task to broadcast metrics to all connected clients"""
    while True:
        try:
            if websocket_connections:
                metrics = await get_realtime_metrics()
                
                message = json.dumps({
                    "type": "metrics_update",
                    "data": metrics
                })
                
                # Broadcast to all connected clients
                disconnected = []
                for ws in websocket_connections:
                    try:
                        await ws.send_text(message)
                    except:
                        disconnected.append(ws)
                
                # Remove disconnected clients
                for ws in disconnected:
                    websocket_connections.remove(ws)
            
            await asyncio.sleep(2)  # Update every 2 seconds
        except Exception as e:
            print(f"Error broadcasting metrics: {e}")
            await asyncio.sleep(5)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
