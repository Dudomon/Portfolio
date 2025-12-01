# API Reference

## Base URL

```
http://localhost:8080
```

## Authentication

Currently, the API does not require authentication. In production, implement API key or JWT authentication.

---

## Endpoints

### Health Check

#### `GET /`

Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "Real-Time Analytics API",
  "version": "1.0.0",
  "timestamp": "2024-11-30T10:30:00.000Z"
}
```

#### `GET /health`

Detailed health check for all services.

**Response:**
```json
{
  "api": "healthy",
  "clickhouse": "healthy",
  "kafka": "healthy",
  "redis": "healthy"
}
```

---

### Events

#### `POST /events`

Ingest a single event into the pipeline.

**Request Body:**
```json
{
  "event_id": "evt_abc123",
  "event_type": "purchase",
  "timestamp": "2024-11-30T10:30:00.000Z",
  "user_id": "user_001",
  "session_id": "sess_xyz",
  "product_id": "prod_001",
  "product_name": "Laptop Pro",
  "category": "Electronics",
  "price": 1299.99,
  "quantity": 1,
  "revenue": 1299.99,
  "country": "USA",
  "city": "New York",
  "device_type": "desktop",
  "browser": "Chrome",
  "metadata": {}
}
```

**Response:**
```json
{
  "status": "success",
  "event_id": "evt_abc123"
}
```

#### `POST /events/batch`

Ingest multiple events in a single request.

**Request Body:**
```json
[
  {
    "event_id": "evt_001",
    "event_type": "purchase",
    ...
  },
  {
    "event_id": "evt_002",
    "event_type": "view",
    ...
  }
]
```

**Response:**
```json
{
  "status": "success",
  "count": 2
}
```

---

### Metrics

#### `GET /metrics/realtime`

Get real-time metrics for the last minute.

**Response:**
```json
{
  "total_events": 1523,
  "unique_users": 847,
  "total_revenue": 45678.90,
  "avg_order_value": 89.99,
  "timestamp": "2024-11-30T10:30:00.000Z"
}
```

#### `GET /metrics/timeseries`

Get time-series metrics over a specified window.

**Query Parameters:**
- `window` (optional): Time window - `1m`, `5m`, `15m`, `1h`, `24h` (default: `1h`)
- `metric` (optional): Specific metric - `revenue`, `events`, `users` (default: `revenue`)

**Example:**
```
GET /metrics/timeseries?window=1h&metric=revenue
```

**Response:**
```json
{
  "data": [
    {
      "timestamp": "2024-11-30T09:00:00.000Z",
      "events": 1200,
      "revenue": 34567.89,
      "users": 456
    },
    {
      "timestamp": "2024-11-30T10:00:00.000Z",
      "events": 1523,
      "revenue": 45678.90,
      "users": 567
    }
  ],
  "window": "1h"
}
```

---

### Products

#### `GET /products/top`

Get top-selling products.

**Query Parameters:**
- `limit` (optional): Number of products to return (default: `10`)

**Example:**
```
GET /products/top?limit=5
```

**Response:**
```json
{
  "products": [
    {
      "product_id": "prod_001",
      "name": "Laptop Pro",
      "category": "Electronics",
      "sales_count": 234,
      "total_revenue": 304197.66
    },
    {
      "product_id": "prod_002",
      "name": "Wireless Mouse",
      "category": "Accessories",
      "sales_count": 567,
      "total_revenue": 16997.33
    }
  ]
}
```

---

### Geographic Distribution

#### `GET /geo/distribution`

Get geographic distribution of sales.

**Response:**
```json
{
  "distribution": [
    {
      "country": "USA",
      "events": 5234,
      "revenue": 156789.45,
      "users": 2345
    },
    {
      "country": "Brazil",
      "events": 3456,
      "revenue": 98765.43,
      "users": 1567
    }
  ]
}
```

---

### Alerts

#### `GET /alerts`

Get recent alerts.

**Query Parameters:**
- `limit` (optional): Number of alerts to return (default: `50`)

**Example:**
```
GET /alerts?limit=20
```

**Response:**
```json
{
  "alerts": [
    {
      "alert_id": "alert_001",
      "alert_type": "high_transaction_rate",
      "severity": "high",
      "timestamp": "2024-11-30T10:25:00.000Z",
      "metric_name": "transactions_per_second",
      "metric_value": 5234.56,
      "threshold": 5000.00,
      "message": "Transaction rate exceeded threshold",
      "resolved": false
    }
  ]
}
```

---

## WebSocket

### `WS /ws`

WebSocket endpoint for real-time metric updates.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = () => {
  console.log('Connected');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

**Message Format:**
```json
{
  "type": "metrics_update",
  "data": {
    "total_events": 1523,
    "unique_users": 847,
    "total_revenue": 45678.90,
    "avg_order_value": 89.99,
    "timestamp": "2024-11-30T10:30:00.000Z"
  }
}
```

**Update Frequency:** Every 2 seconds

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid request body"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Database connection failed"
}
```

---

## Rate Limiting

Currently, no rate limiting is implemented. For production:
- Implement rate limiting per IP/API key
- Suggested: 1000 requests per minute per client
- Use Redis for distributed rate limiting

---

## CORS

CORS is enabled for:
- `http://localhost:3000` (development)

Update `CORS_ORIGINS` environment variable for production domains.

---

## Examples

### Python

```python
import requests

# Send event
response = requests.post(
    'http://localhost:8080/events',
    json={
        'event_id': 'evt_001',
        'event_type': 'purchase',
        'user_id': 'user_001',
        'product_id': 'prod_001',
        'revenue': 99.99
    }
)

# Get metrics
metrics = requests.get('http://localhost:8080/metrics/realtime').json()
print(metrics)
```

### JavaScript

```javascript
// Send event
fetch('http://localhost:8080/events', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    event_id: 'evt_001',
    event_type: 'purchase',
    user_id: 'user_001',
    product_id: 'prod_001',
    revenue: 99.99
  })
});

// Get metrics
const metrics = await fetch('http://localhost:8080/metrics/realtime')
  .then(res => res.json());
console.log(metrics);
```

### cURL

```bash
# Send event
curl -X POST http://localhost:8080/events \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "evt_001",
    "event_type": "purchase",
    "user_id": "user_001",
    "product_id": "prod_001",
    "revenue": 99.99
  }'

# Get metrics
curl http://localhost:8080/metrics/realtime
```
