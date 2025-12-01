# API Reference

## Base URL

```
http://ecommerce.local/api
```

## Order Service API

### Health Check

#### GET /orders/health

Check service health status.

**Response:**
```json
{
  "status": "healthy",
  "service": "order-service",
  "version": "1.0.0"
}
```

### Readiness Check

#### GET /orders/ready

Check if service is ready to accept requests.

**Response:**
```json
{
  "status": "ready"
}
```

### Create Order

#### POST /orders/orders

Create a new order and initiate the saga pattern for distributed transaction.

**Request Body:**
```json
{
  "user_id": "user_123",
  "items": [
    {
      "product_id": "prod_001",
      "product_name": "Laptop Pro",
      "quantity": 1,
      "price": 1299.99
    },
    {
      "product_id": "prod_002",
      "product_name": "Wireless Mouse",
      "quantity": 2,
      "price": 29.99
    }
  ]
}
```

**Response:** `201 Created`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user_123",
  "items": [...],
  "total_amount": 1359.97,
  "status": "pending",
  "created_at": "2024-11-30T10:00:00Z",
  "updated_at": "2024-11-30T10:00:00Z"
}
```

### Get Order

#### GET /orders/orders/{order_id}

Retrieve order details by ID.

**Path Parameters:**
- `order_id` (string, required): Order UUID

**Response:** `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user_123",
  "items": [...],
  "total_amount": 1359.97,
  "status": "completed",
  "created_at": "2024-11-30T10:00:00Z",
  "updated_at": "2024-11-30T10:05:00Z"
}
```

**Error Response:** `404 Not Found`
```json
{
  "detail": "Order not found"
}
```

### List Orders

#### GET /orders/orders

List orders with optional filters.

**Query Parameters:**
- `user_id` (string, optional): Filter by user ID
- `status` (string, optional): Filter by status (pending, completed, cancelled, failed)
- `skip` (integer, optional): Number of records to skip (default: 0)
- `limit` (integer, optional): Maximum number of records (default: 100)

**Response:** `200 OK`
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "user_123",
    "total_amount": 1359.97,
    "status": "completed",
    "created_at": "2024-11-30T10:00:00Z"
  }
]
```

### Update Order Status

#### PATCH /orders/orders/{order_id}/status

Update order status manually.

**Path Parameters:**
- `order_id` (string, required): Order UUID

**Query Parameters:**
- `status` (string, required): New status

**Response:** `200 OK`
```json
{
  "status": "updated"
}
```

### Get Order Events

#### GET /orders/orders/{order_id}/events

Retrieve event history for an order (Event Sourcing).

**Path Parameters:**
- `order_id` (string, required): Order UUID

**Response:** `200 OK`
```json
[
  {
    "id": "event_001",
    "order_id": "550e8400-e29b-41d4-a716-446655440000",
    "event_type": "order_created",
    "event_data": {...},
    "created_at": "2024-11-30T10:00:00Z"
  },
  {
    "id": "event_002",
    "order_id": "550e8400-e29b-41d4-a716-446655440000",
    "event_type": "payment_completed",
    "event_data": {...},
    "created_at": "2024-11-30T10:01:00Z"
  }
]
```

### Get Metrics

#### GET /orders/metrics

Get service metrics and statistics.

**Response:** `200 OK`
```json
{
  "total_orders": 1523,
  "pending_orders": 45,
  "completed_orders": 1420,
  "failed_orders": 58
}
```

## Event Bus Messages

### Published Events

#### order.created
Published when a new order is created.

```json
{
  "order_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user_123",
  "items": [...],
  "total_amount": 1359.97,
  "timestamp": "2024-11-30T10:00:00Z"
}
```

#### order.status_changed
Published when order status changes.

```json
{
  "order_id": "550e8400-e29b-41d4-a716-446655440000",
  "old_status": "pending",
  "new_status": "completed",
  "timestamp": "2024-11-30T10:05:00Z"
}
```

#### order.completed
Published when order is successfully completed.

```json
{
  "order_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user_123",
  "total_amount": 1359.97
}
```

#### order.cancelled
Published when order is cancelled.

```json
{
  "order_id": "550e8400-e29b-41d4-a716-446655440000",
  "reason": "payment_failed"
}
```

### Consumed Events

#### payment.completed
Consumed from Payment Service.

```json
{
  "order_id": "550e8400-e29b-41d4-a716-446655440000",
  "payment_id": "pay_123",
  "amount": 1359.97,
  "timestamp": "2024-11-30T10:01:00Z"
}
```

#### payment.failed
Consumed from Payment Service.

```json
{
  "order_id": "550e8400-e29b-41d4-a716-446655440000",
  "reason": "insufficient_funds",
  "timestamp": "2024-11-30T10:01:00Z"
}
```

#### inventory.reserved
Consumed from Product Service.

```json
{
  "order_id": "550e8400-e29b-41d4-a716-446655440000",
  "items": [...],
  "timestamp": "2024-11-30T10:02:00Z"
}
```

#### inventory.failed
Consumed from Product Service.

```json
{
  "order_id": "550e8400-e29b-41d4-a716-446655440000",
  "reason": "out_of_stock",
  "timestamp": "2024-11-30T10:02:00Z"
}
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid request body"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Service not ready"
}
```

## Rate Limiting

Currently not implemented. For production:
- Recommended: 1000 requests per minute per client
- Implement using API Gateway (Kong)
- Use Redis for distributed rate limiting

## Authentication

Currently not implemented. For production:
- JWT tokens recommended
- Implement in API Gateway
- Service-to-service authentication with mTLS

## Versioning

API versioning strategy:
- URL path versioning: `/api/v1/orders`
- Header versioning: `Accept: application/vnd.ecommerce.v1+json`

## Examples

### cURL

```bash
# Create order
curl -X POST http://ecommerce.local/api/orders/orders \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "items": [
      {
        "product_id": "prod_001",
        "product_name": "Laptop",
        "quantity": 1,
        "price": 1299.99
      }
    ]
  }'

# Get order
curl http://ecommerce.local/api/orders/orders/{order_id}

# List orders
curl "http://ecommerce.local/api/orders/orders?user_id=user_123&limit=10"
```

### Python

```python
import requests

# Create order
response = requests.post(
    'http://ecommerce.local/api/orders/orders',
    json={
        'user_id': 'user_123',
        'items': [
            {
                'product_id': 'prod_001',
                'product_name': 'Laptop',
                'quantity': 1,
                'price': 1299.99
            }
        ]
    }
)

order = response.json()
print(f"Order created: {order['id']}")

# Get order
order_id = order['id']
response = requests.get(f'http://ecommerce.local/api/orders/orders/{order_id}')
print(response.json())
```

### JavaScript

```javascript
// Create order
const response = await fetch('http://ecommerce.local/api/orders/orders', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    user_id: 'user_123',
    items: [
      {
        product_id: 'prod_001',
        product_name: 'Laptop',
        quantity: 1,
        price: 1299.99
      }
    ]
  })
});

const order = await response.json();
console.log('Order created:', order.id);

// Get order
const orderResponse = await fetch(`http://ecommerce.local/api/orders/orders/${order.id}`);
const orderData = await orderResponse.json();
console.log(orderData);
```
