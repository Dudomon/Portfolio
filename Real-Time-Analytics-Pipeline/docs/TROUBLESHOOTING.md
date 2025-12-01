# Troubleshooting Guide

## Common Issues

### 1. Services Not Starting

#### Symptom
```bash
docker-compose up -d
# Services fail to start or restart continuously
```

#### Solutions

**Check logs:**
```bash
docker-compose logs -f [service-name]
```

**Verify ports are available:**
```bash
# Windows
netstat -ano | findstr "8080 9092 8123 6379 3000"

# Linux/Mac
lsof -i :8080,9092,8123,6379,3000
```

**Increase Docker resources:**
- Docker Desktop → Settings → Resources
- Minimum: 8GB RAM, 4 CPUs

**Clean and restart:**
```bash
docker-compose down -v
docker-compose up -d
```

---

### 2. Kafka Connection Issues

#### Symptom
```
Error: Failed to connect to Kafka broker
```

#### Solutions

**Verify Kafka is running:**
```bash
docker-compose ps kafka
```

**Check Kafka logs:**
```bash
docker-compose logs kafka
```

**Test Kafka connectivity:**
```bash
docker exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092
```

**Create topics manually:**
```bash
docker exec kafka kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic ecommerce-events \
  --partitions 10 \
  --replication-factor 1
```

---

### 3. ClickHouse Connection Failed

#### Symptom
```
Error: ClickHouse connection refused
```

#### Solutions

**Check ClickHouse status:**
```bash
docker-compose ps clickhouse
curl http://localhost:8123/ping
```

**Verify credentials:**
```bash
docker exec clickhouse clickhouse-client \
  --user admin \
  --password admin123 \
  --query "SELECT 1"
```

**Check database initialization:**
```bash
docker exec clickhouse clickhouse-client \
  --user admin \
  --password admin123 \
  --query "SHOW DATABASES"
```

**Reinitialize database:**
```bash
docker-compose down clickhouse
docker volume rm real-time-analytics-pipeline_clickhouse-data
docker-compose up -d clickhouse
```

---

### 4. Flink Job Not Running

#### Symptom
```
Flink dashboard shows no running jobs
```

#### Solutions

**Access Flink dashboard:**
```
http://localhost:8081
```

**Check TaskManager status:**
```bash
docker-compose ps flink-taskmanager
docker-compose logs flink-taskmanager
```

**Submit job manually:**
```bash
docker exec flink-jobmanager flink run \
  -py /opt/flink/jobs/stream_processor.py
```

**Increase parallelism:**
Edit `docker-compose.yml`:
```yaml
flink-taskmanager:
  environment:
    - taskmanager.numberOfTaskSlots: 8
  scale: 3
```

---

### 5. Dashboard Not Loading

#### Symptom
```
Cannot access http://localhost:3000
```

#### Solutions

**Check React app logs:**
```bash
docker-compose logs dashboard
```

**Verify API connection:**
```bash
curl http://localhost:8080/health
```

**Rebuild dashboard:**
```bash
docker-compose down dashboard
docker-compose build --no-cache dashboard
docker-compose up -d dashboard
```

**Check environment variables:**
```bash
docker exec dashboard env | grep REACT_APP
```

---

### 6. WebSocket Connection Failed

#### Symptom
```
Dashboard shows "Disconnected" status
```

#### Solutions

**Test WebSocket manually:**
```javascript
// Browser console
const ws = new WebSocket('ws://localhost:8080/ws');
ws.onopen = () => console.log('Connected');
ws.onerror = (e) => console.error('Error:', e);
```

**Check API WebSocket endpoint:**
```bash
docker-compose logs api | grep -i websocket
```

**Verify CORS settings:**
Edit `api/config.py`:
```python
CORS_ORIGINS = ["http://localhost:3000"]
```

---

### 7. High Memory Usage

#### Symptom
```
Docker containers consuming too much memory
```

#### Solutions

**Check memory usage:**
```bash
docker stats
```

**Reduce Flink state size:**
Edit `docker-compose.yml`:
```yaml
flink-jobmanager:
  environment:
    - taskmanager.memory.process.size: 2048m
```

**Enable ClickHouse compression:**
```sql
ALTER TABLE events MODIFY SETTING
  storage_policy = 'default',
  compress_method = 'lz4';
```

**Adjust Redis maxmemory:**
```bash
docker exec redis redis-cli CONFIG SET maxmemory 512mb
docker exec redis redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

---

### 8. Slow Query Performance

#### Symptom
```
API responses taking >5 seconds
```

#### Solutions

**Check ClickHouse query log:**
```sql
SELECT
    query,
    query_duration_ms,
    read_rows,
    read_bytes
FROM system.query_log
WHERE type = 'QueryFinish'
ORDER BY query_duration_ms DESC
LIMIT 10;
```

**Add indexes:**
```sql
ALTER TABLE events ADD INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1;
```

**Enable Redis caching:**
Verify cache is working:
```bash
docker exec redis redis-cli KEYS "metrics:*"
```

**Optimize materialized views:**
```sql
OPTIMIZE TABLE mv_realtime_metrics FINAL;
```

---

### 9. Event Producer Not Sending

#### Symptom
```
python scripts/generate_events.py
# No events appearing in dashboard
```

#### Solutions

**Check API connectivity:**
```bash
curl -X POST http://localhost:8080/events \
  -H "Content-Type: application/json" \
  -d '{"event_id":"test","event_type":"purchase","user_id":"test"}'
```

**Verify Kafka topic:**
```bash
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic ecommerce-events \
  --from-beginning \
  --max-messages 10
```

**Check producer logs:**
```bash
python scripts/generate_events.py --api-url http://localhost:8080 --rate 10
```

---

### 10. Data Not Appearing in Dashboard

#### Symptom
```
Events sent but dashboard shows zero metrics
```

#### Solutions

**Verify data in ClickHouse:**
```bash
docker exec clickhouse clickhouse-client \
  --user admin \
  --password admin123 \
  --query "SELECT count() FROM analytics.events"
```

**Check Flink processing:**
```bash
# Access Flink dashboard
open http://localhost:8081
# Check job metrics and backpressure
```

**Verify API queries:**
```bash
curl http://localhost:8080/metrics/realtime
```

**Check WebSocket updates:**
```bash
# Browser console
const ws = new WebSocket('ws://localhost:8080/ws');
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

---

## Performance Tuning

### Kafka

```properties
# Increase throughput
batch.size=32768
linger.ms=10
compression.type=snappy
```

### Flink

```yaml
# Increase checkpoint interval
execution.checkpointing.interval: 120000

# Tune state backend
state.backend.rocksdb.block.cache-size: 512m
```

### ClickHouse

```sql
-- Optimize table
OPTIMIZE TABLE events FINAL;

-- Adjust merge settings
ALTER TABLE events MODIFY SETTING
  max_bytes_to_merge_at_max_space_in_pool = 161061273600;
```

---

## Monitoring Commands

### Check all services
```bash
docker-compose ps
```

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Last 100 lines
docker-compose logs --tail=100 kafka
```

### Resource usage
```bash
docker stats
```

### Network connectivity
```bash
docker network inspect real-time-analytics-pipeline_analytics-network
```

---

## Getting Help

1. **Check logs first**: Most issues are visible in logs
2. **GitHub Issues**: Report bugs with logs and steps to reproduce
3. **Documentation**: Review architecture and API docs
4. **Community**: Join discussions for help

---

## Clean Slate

If all else fails, complete reset:

```bash
# Stop all services
docker-compose down -v

# Remove all containers
docker-compose rm -f

# Remove volumes
docker volume prune -f

# Rebuild and restart
docker-compose build --no-cache
docker-compose up -d

# Wait for services to be healthy
docker-compose ps
```
