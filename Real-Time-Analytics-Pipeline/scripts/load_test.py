"""
Load Testing Script
Tests the pipeline under high load conditions
"""

import argparse
import asyncio
import aiohttp
import time
import random
import uuid
from datetime import datetime
from statistics import mean, median, stdev


class LoadTester:
    def __init__(self, api_url, rate, duration, batch_size=10):
        self.api_url = api_url
        self.rate = rate
        self.duration = duration
        self.batch_size = batch_size
        self.results = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'latencies': [],
            'errors': []
        }
    
    def generate_event(self):
        """Generate a random event"""
        products = [
            {"id": "prod_001", "name": "Laptop Pro", "price": 1299.99},
            {"id": "prod_002", "name": "Mouse", "price": 29.99},
            {"id": "prod_003", "name": "Keyboard", "price": 149.99},
        ]
        
        product = random.choice(products)
        
        return {
            "event_id": f"evt_{uuid.uuid4().hex[:12]}",
            "event_type": random.choice(["purchase", "view", "cart_add"]),
            "timestamp": datetime.now().isoformat(),
            "user_id": f"user_{random.randint(1, 10000):05d}",
            "session_id": f"sess_{uuid.uuid4().hex[:12]}",
            "product_id": product["id"],
            "product_name": product["name"],
            "category": "Electronics",
            "price": product["price"],
            "quantity": random.randint(1, 3),
            "revenue": product["price"] * random.randint(1, 3),
            "country": random.choice(["USA", "Brazil", "Germany"]),
            "city": "Test City",
            "device_type": random.choice(["desktop", "mobile"]),
            "browser": "Chrome",
            "metadata": {}
        }
    
    async def send_batch(self, session):
        """Send a batch of events"""
        batch = [self.generate_event() for _ in range(self.batch_size)]
        
        start_time = time.time()
        try:
            async with session.post(
                f"{self.api_url}/events/batch",
                json=batch,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                latency = (time.time() - start_time) * 1000  # ms
                
                if response.status == 200:
                    self.results['successful'] += len(batch)
                    self.results['latencies'].append(latency)
                else:
                    self.results['failed'] += len(batch)
                    self.results['errors'].append(f"HTTP {response.status}")
                
                self.results['total_requests'] += len(batch)
        
        except Exception as e:
            self.results['failed'] += len(batch)
            self.results['errors'].append(str(e))
            self.results['total_requests'] += len(batch)
    
    async def run_load_test(self):
        """Run the load test"""
        print(f"🚀 Starting load test")
        print(f"   Target rate: {self.rate} events/sec")
        print(f"   Duration: {self.duration}s")
        print(f"   Batch size: {self.batch_size}")
        print()
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            batches_per_second = self.rate // self.batch_size
            interval = 1.0 / batches_per_second if batches_per_second > 0 else 1.0
            
            while (time.time() - start_time) < self.duration:
                batch_start = time.time()
                
                # Create batch tasks
                task = asyncio.create_task(self.send_batch(session))
                tasks.append(task)
                
                # Print progress every 10 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                    self.print_progress(elapsed)
                
                # Wait to maintain rate
                elapsed_batch = time.time() - batch_start
                sleep_time = max(0, interval - elapsed_batch)
                await asyncio.sleep(sleep_time)
            
            # Wait for remaining tasks
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.print_results()
    
    def print_progress(self, elapsed):
        """Print progress update"""
        actual_rate = self.results['total_requests'] / elapsed if elapsed > 0 else 0
        success_rate = (self.results['successful'] / self.results['total_requests'] * 100) if self.results['total_requests'] > 0 else 0
        
        print(f"⏱️  {int(elapsed)}s | "
              f"Sent: {self.results['total_requests']:,} | "
              f"Rate: {actual_rate:.1f}/s | "
              f"Success: {success_rate:.1f}%")
    
    def print_results(self):
        """Print final results"""
        print("\n" + "="*60)
        print("📊 LOAD TEST RESULTS")
        print("="*60)
        
        total_time = self.duration
        actual_rate = self.results['total_requests'] / total_time if total_time > 0 else 0
        success_rate = (self.results['successful'] / self.results['total_requests'] * 100) if self.results['total_requests'] > 0 else 0
        
        print(f"\n📈 Throughput:")
        print(f"   Total requests: {self.results['total_requests']:,}")
        print(f"   Successful: {self.results['successful']:,}")
        print(f"   Failed: {self.results['failed']:,}")
        print(f"   Success rate: {success_rate:.2f}%")
        print(f"   Actual rate: {actual_rate:.1f} events/sec")
        
        if self.results['latencies']:
            print(f"\n⏱️  Latency (ms):")
            print(f"   Mean: {mean(self.results['latencies']):.2f}")
            print(f"   Median: {median(self.results['latencies']):.2f}")
            print(f"   Min: {min(self.results['latencies']):.2f}")
            print(f"   Max: {max(self.results['latencies']):.2f}")
            if len(self.results['latencies']) > 1:
                print(f"   Std Dev: {stdev(self.results['latencies']):.2f}")
            
            # Percentiles
            sorted_latencies = sorted(self.results['latencies'])
            p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
            p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
            
            print(f"\n📊 Percentiles:")
            print(f"   P50: {p50:.2f} ms")
            print(f"   P95: {p95:.2f} ms")
            print(f"   P99: {p99:.2f} ms")
        
        if self.results['errors']:
            print(f"\n❌ Errors ({len(self.results['errors'])}):")
            error_counts = {}
            for error in self.results['errors'][:10]:  # Show first 10
                error_counts[error] = error_counts.get(error, 0) + 1
            
            for error, count in error_counts.items():
                print(f"   {error}: {count}")
        
        print("\n" + "="*60)
        
        # Performance assessment
        if success_rate >= 99 and p99 < 1000:
            print("✅ EXCELLENT: System performing well under load")
        elif success_rate >= 95 and p99 < 2000:
            print("⚠️  GOOD: System handling load with acceptable performance")
        elif success_rate >= 90:
            print("⚠️  WARNING: System showing signs of stress")
        else:
            print("❌ CRITICAL: System unable to handle load")


def main():
    parser = argparse.ArgumentParser(description="Load test the analytics pipeline")
    parser.add_argument("--api-url", default="http://localhost:8080", help="API URL")
    parser.add_argument("--rate", type=int, default=1000, help="Target events per second")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--batch-size", type=int, default=10, help="Events per batch")
    
    args = parser.parse_args()
    
    tester = LoadTester(
        api_url=args.api_url,
        rate=args.rate,
        duration=args.duration,
        batch_size=args.batch_size
    )
    
    asyncio.run(tester.run_load_test())


if __name__ == "__main__":
    main()
