# ⚡ Quick Start Guide

Get the Real-Time Analytics Pipeline running in 5 minutes!

## Prerequisites

- Docker Desktop (8GB RAM minimum)
- Python 3.8+ (for event generator)
- Ports available: 8080, 8081, 9092, 8123, 6379, 3000

## Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/real-time-analytics-pipeline.git
cd real-time-analytics-pipeline
```

## Step 2: Start Services

### Windows
```bash
setup.bat
```

### Linux/Mac
```bash
chmod +x setup.sh
./setup.sh
```

### Manual
```bash
docker-compose up -d
```

## Step 3: Verify Services

Check all services are running:
```bash
docker-compose ps
```

You should see:
- ✅ zookeeper
- ✅ kafka
- ✅ clickhouse
- ✅ redis
- ✅ flink-jobmanager
- ✅ flink-taskmanager
- ✅ api
- ✅ dashboard

## Step 4: Access Dashboard

Open your browser:
```
http://localhost:3000
```

## Step 5: Generate Events

In a new terminal:
```bash
# Install dependencies
pip install requests

# Generate 1000 events/second for 60 seconds
python scripts/generate_events.py --rate 1000 --duration 60
```

## Step 6: Watch Real-Time Updates

The dashboard will automatically update every 2 seconds showing:
- 📈 Total events
- 👥 Unique users
- 💰 Total revenue
- 🛒 Average order value
- 📊 Time-series charts
- 🏆 Top products
- 🌍 Geographic distribution

## Troubleshooting

### Services not starting?
```bash
# Check logs
docker-compose logs -f

# Restart services
docker-compose restart
```

### Dashboard not loading?
```bash
# Check API health
curl http://localhost:8080/health

# Rebuild dashboard
docker-compose build dashboard
docker-compose up -d dashboard
```

### No data appearing?
```bash
# Verify events in ClickHouse
docker exec clickhouse clickhouse-client --user admin --password admin123 --query "SELECT count() FROM analytics.events"

# Check Kafka topic
docker exec kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic ecommerce-events --from-beginning --max-messages 5
```

## Next Steps

- 📚 Read [ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design
- 🔧 Check [API.md](docs/API.md) for API reference
- 🚀 See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for production setup
- 🐛 Visit [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues

## Stopping Services

```bash
docker-compose down
```

To remove all data:
```bash
docker-compose down -v
```

## Support

Having issues? Check:
1. Docker has enough resources (8GB RAM minimum)
2. All required ports are available
3. Docker daemon is running
4. Logs for error messages: `docker-compose logs -f`

---

**Enjoy your real-time analytics pipeline! 🚀**
