#!/bin/bash

# API Testing Script
# Tests all microservices endpoints

set -e

API_URL="${API_URL:-http://ecommerce.local}"
ORDER_SERVICE="${API_URL}/api/orders"

echo "Testing E-Commerce Microservices API"
echo "======================================"
echo ""

# Test Order Service Health
echo "1. Testing Order Service Health..."
curl -s "${ORDER_SERVICE}/health" | jq .
echo ""

# Create Order
echo "2. Creating new order..."
ORDER_RESPONSE=$(curl -s -X POST "${ORDER_SERVICE}/orders" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "items": [
      {
        "product_id": "prod_001",
        "product_name": "Laptop",
        "quantity": 1,
        "price": 1299.99
      },
      {
        "product_id": "prod_002",
        "product_name": "Mouse",
        "quantity": 2,
        "price": 29.99
      }
    ]
  }')

echo "$ORDER_RESPONSE" | jq .
ORDER_ID=$(echo "$ORDER_RESPONSE" | jq -r '.id')
echo "Order ID: $ORDER_ID"
echo ""

# Get Order
echo "3. Retrieving order..."
curl -s "${ORDER_SERVICE}/orders/${ORDER_ID}" | jq .
echo ""

# List Orders
echo "4. Listing all orders..."
curl -s "${ORDER_SERVICE}/orders?limit=5" | jq .
echo ""

# Get Order Events
echo "5. Getting order events (Event Sourcing)..."
curl -s "${ORDER_SERVICE}/orders/${ORDER_ID}/events" | jq .
echo ""

# Get Metrics
echo "6. Getting service metrics..."
curl -s "${ORDER_SERVICE}/metrics" | jq .
echo ""

echo "======================================"
echo "API Testing Complete!"
