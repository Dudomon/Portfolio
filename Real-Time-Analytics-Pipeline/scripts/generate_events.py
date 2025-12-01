"""
Event Generator Script
Simulates e-commerce events and sends them to Kafka
"""

import argparse
import random
import time
import uuid
from datetime import datetime
from decimal import Decimal
import requests
import json


# Sample data
PRODUCTS = [
    {"id": "prod_001", "name": "Laptop Pro 15", "category": "Electronics", "price": 1299.99},
    {"id": "prod_002", "name": "Wireless Mouse", "category": "Accessories", "price": 29.99},
    {"id": "prod_003", "name": "Mechanical Keyboard", "category": "Accessories", "price": 149.99},
    {"id": "prod_004", "name": "USB-C Cable", "category": "Accessories", "price": 19.99},
    {"id": "prod_005", "name": "4K Monitor", "category": "Electronics", "price": 599.99},
    {"id": "prod_006", "name": "Webcam HD", "category": "Electronics", "price": 89.99},
    {"id": "prod_007", "name": "Desk Lamp", "category": "Office", "price": 39.99},
    {"id": "prod_008", "name": "Office Chair", "category": "Furniture", "price": 299.99},
    {"id": "prod_009", "name": "Standing Desk", "category": "Furniture", "price": 499.99},
    {"id": "prod_010", "name": "Headphones", "category": "Electronics", "price": 199.99},
]

COUNTRIES = ["USA", "Brazil", "Germany", "UK", "Canada", "France", "Japan", "Australia"]
CITIES = {
    "USA": ["New York", "Los Angeles", "Chicago", "Houston"],
    "Brazil": ["São Paulo", "Rio de Janeiro", "Brasília", "Salvador"],
    "Germany": ["Berlin", "Munich", "Hamburg", "Frankfurt"],
    "UK": ["London", "Manchester", "Birmingham", "Leeds"],
    "Canada": ["Toronto", "Vancouver", "Montreal", "Calgary"],
    "France": ["Paris", "Lyon", "Marseille", "Toulouse"],
    "Japan": ["Tokyo", "Osaka", "Kyoto", "Yokohama"],
    "Australia": ["Sydney", "Melbourne", "Brisbane", "Perth"]
}

DEVICE_TYPES = ["desktop", "mobile", "tablet"]
BROWSERS = ["Chrome", "Firefox", "Safari", "Edge"]
EVENT_TYPES = ["view", "cart_add", "purchase"]


def generate_event():
    """Generate a random e-commerce event"""
    event_type = random.choices(
        EVENT_TYPES,
        weights=[60, 25, 15],  # More views, fewer purchases
        k=1
    )[0]
    
    product = random.choice(PRODUCTS)
    country = random.choice(COUNTRIES)
    city = random.choice(CITIES[country])
    
    quantity = random.randint(1, 3) if event_type == "purchase" else 0
    revenue = float(product["price"]) * quantity if event_type == "purchase" else 0
    
    event = {
        "event_id": f"evt_{uuid.uuid4().hex[:12]}",
        "event_type": event_type,
        "timestamp": datetime.now().isoformat(),
        "user_id": f"user_{random.randint(1, 10000):05d}",
        "session_id": f"sess_{uuid.uuid4().hex[:12]}",
        "product_id": product["id"],
        "product_name": product["name"],
        "category": product["category"],
        "price": product["price"],
        "quantity": quantity,
        "revenue": revenue,
        "country": country,
        "city": city,
        "device_type": random.choice(DEVICE_TYPES),
        "browser": random.choice(BROWSERS),
        "metadata": {}
    }
    
    return event


def send_event(api_url, event):
    """Send event to API"""
    try:
        response = requests.post(
            f"{api_url}/events",
            json=event,
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending event: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate e-commerce events")
    parser.add_argument("--api-url", default="http://localhost:8080", help="API URL")
    parser.add_argument("--rate", type=int, default=100, help="Events per second")
    parser.add_argument("--duration", type=int, default=0, help="Duration in seconds (0 = infinite)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for sending events")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting event generator")
    print(f"   API URL: {args.api_url}")
    print(f"   Rate: {args.rate} events/second")
    print(f"   Duration: {'infinite' if args.duration == 0 else f'{args.duration}s'}")
    print(f"   Batch size: {args.batch_size}")
    print()
    
    events_sent = 0
    errors = 0
    start_time = time.time()
    
    try:
        while True:
            batch_start = time.time()
            
            # Generate and send batch
            batch = [generate_event() for _ in range(args.batch_size)]
            
            try:
                response = requests.post(
                    f"{args.api_url}/events/batch",
                    json=batch,
                    timeout=5
                )
                if response.status_code == 200:
                    events_sent += len(batch)
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                print(f"❌ Error: {e}")
            
            # Calculate sleep time to maintain rate
            elapsed = time.time() - batch_start
            target_time = args.batch_size / args.rate
            sleep_time = max(0, target_time - elapsed)
            time.sleep(sleep_time)
            
            # Print stats every 10 seconds
            if events_sent % (args.rate * 10) == 0:
                elapsed_total = time.time() - start_time
                actual_rate = events_sent / elapsed_total if elapsed_total > 0 else 0
                print(f"📊 Sent: {events_sent:,} events | Rate: {actual_rate:.1f}/s | Errors: {errors}")
            
            # Check duration
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                break
    
    except KeyboardInterrupt:
        print("\n⏹️  Stopping event generator...")
    
    finally:
        elapsed_total = time.time() - start_time
        actual_rate = events_sent / elapsed_total if elapsed_total > 0 else 0
        print(f"\n✅ Summary:")
        print(f"   Total events: {events_sent:,}")
        print(f"   Duration: {elapsed_total:.1f}s")
        print(f"   Average rate: {actual_rate:.1f} events/s")
        print(f"   Errors: {errors}")


if __name__ == "__main__":
    main()
