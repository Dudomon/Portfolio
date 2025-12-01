# Deployment Guide

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB RAM minimum (16GB recommended)
- 20GB free disk space

## Local Development

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/real-time-analytics-pipeline.git
cd real-time-analytics-pipeline
```

### 2. Start Services

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 3. Verify Services

```bash
# Check API health
curl http://localhost:8080/health

# Check Kafka
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Check ClickHouse
curl http://localhost:8123/ping

# Check Redis
docker exec redis redis-cli ping
```

### 4. Access Dashboard

Open browser: http://localhost:3000

### 5. Generate Sample Data

```bash
# Install Python dependencies
pip install requests

# Generate events (1000 events/second for 60 seconds)
python scripts/generate_events.py --rate 1000 --duration 60
```

## Production Deployment

### Kubernetes

#### 1. Create Namespace

```bash
kubectl create namespace analytics
```

#### 2. Deploy Kafka

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install kafka bitnami/kafka \
  --namespace analytics \
  --set replicaCount=3 \
  --set persistence.size=100Gi
```

#### 3. Deploy ClickHouse

```bash
helm repo add clickhouse https://charts.clickhouse.com
helm install clickhouse clickhouse/clickhouse \
  --namespace analytics \
  --set persistence.size=200Gi \
  --set replicaCount=2
```

#### 4. Deploy Redis

```bash
helm install redis bitnami/redis \
  --namespace analytics \
  --set master.persistence.size=10Gi
```

#### 5. Deploy Flink

```bash
helm repo add flink https://apache.github.io/flink-kubernetes-operator
helm install flink flink/flink-kubernetes-operator \
  --namespace analytics
```

#### 6. Deploy API

```bash
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
```

#### 7. Deploy Dashboard

```bash
kubectl apply -f k8s/dashboard-deployment.yaml
kubectl apply -f k8s/dashboard-service.yaml
kubectl apply -f k8s/ingress.yaml
```

### AWS Deployment

#### Architecture

```
Internet → ALB → ECS Fargate
              ↓
         MSK (Kafka)
              ↓
         ECS Flink
              ↓
         RDS ClickHouse / Redshift
              ↓
         ElastiCache (Redis)
```

#### 1. Infrastructure as Code (Terraform)

```hcl
# main.tf
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  # ... configuration
}

module "msk" {
  source = "terraform-aws-modules/msk/aws"
  # ... configuration
}

module "ecs" {
  source = "terraform-aws-modules/ecs/aws"
  # ... configuration
}
```

#### 2. Deploy with Terraform

```bash
terraform init
terraform plan
terraform apply
```

### GCP Deployment

#### Architecture

```
Internet → Cloud Load Balancer → Cloud Run
                                    ↓
                              Pub/Sub
                                    ↓
                              Dataflow
                                    ↓
                              BigQuery
                                    ↓
                              Memorystore (Redis)
```

#### Deploy with gcloud

```bash
# Deploy API
gcloud run deploy analytics-api \
  --image gcr.io/project/analytics-api \
  --platform managed \
  --region us-central1

# Deploy Dashboard
gcloud run deploy analytics-dashboard \
  --image gcr.io/project/analytics-dashboard \
  --platform managed \
  --region us-central1
```

## Configuration

### Environment Variables

Create `.env` file:

```bash
# Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:29092
KAFKA_TOPIC=ecommerce-events

# ClickHouse
CLICKHOUSE_HOST=clickhouse
CLICKHOUSE_PORT=8123
CLICKHOUSE_DB=analytics
CLICKHOUSE_USER=admin
CLICKHOUSE_PASSWORD=your_secure_password

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# API
API_HOST=0.0.0.0
API_PORT=8080
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### Scaling Configuration

#### docker-compose.yml

```yaml
services:
  flink-taskmanager:
    deploy:
      replicas: 4  # Scale to 4 task managers
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

#### Kubernetes

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Monitoring

### Prometheus + Grafana

```bash
# Deploy monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Access Grafana
open http://localhost:3001
```

### Metrics to Monitor

- **Kafka**: Consumer lag, throughput, error rate
- **Flink**: Checkpoint duration, backpressure, state size
- **ClickHouse**: Query duration, merge rate, disk usage
- **API**: Request rate, latency (p50, p95, p99), error rate

## Backup & Recovery

### ClickHouse Backup

```bash
# Backup
clickhouse-client --query "BACKUP DATABASE analytics TO Disk('backups', 'backup_20240101')"

# Restore
clickhouse-client --query "RESTORE DATABASE analytics FROM Disk('backups', 'backup_20240101')"
```

### Kafka Backup

```bash
# Use MirrorMaker 2 for replication
kafka-mirror-maker.sh --consumer.config source.properties \
  --producer.config target.properties \
  --whitelist 'ecommerce-events'
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs service-name

# Check resource usage
docker stats

# Restart service
docker-compose restart service-name
```

### High Latency

```bash
# Check Kafka lag
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group flink-consumer-group --describe

# Check Flink backpressure
# Access Flink UI: http://localhost:8081

# Check ClickHouse slow queries
clickhouse-client --query "SELECT * FROM system.query_log WHERE type='QueryFinish' ORDER BY query_duration_ms DESC LIMIT 10"
```

### Data Loss

```bash
# Check Kafka retention
kafka-configs.sh --bootstrap-server localhost:9092 \
  --entity-type topics --entity-name ecommerce-events --describe

# Check Flink checkpoints
ls -lh /tmp/flink-checkpoints/

# Restore from savepoint
flink run -s /tmp/flink-savepoints/savepoint-xxx job.jar
```

## Security Hardening

### 1. Enable TLS

```yaml
# docker-compose.yml
kafka:
  environment:
    KAFKA_SSL_KEYSTORE_LOCATION: /etc/kafka/secrets/kafka.keystore.jks
    KAFKA_SSL_TRUSTSTORE_LOCATION: /etc/kafka/secrets/kafka.truststore.jks
```

### 2. Enable Authentication

```yaml
clickhouse:
  environment:
    CLICKHOUSE_USER: admin
    CLICKHOUSE_PASSWORD: ${CLICKHOUSE_PASSWORD}
```

### 3. Network Isolation

```yaml
networks:
  analytics-network:
    driver: bridge
    internal: true  # No external access
```

## Performance Tuning

### Kafka

```properties
# server.properties
num.network.threads=8
num.io.threads=16
socket.send.buffer.bytes=102400
socket.receive.buffer.bytes=102400
```

### Flink

```yaml
taskmanager.memory.process.size: 4096m
taskmanager.numberOfTaskSlots: 4
parallelism.default: 4
```

### ClickHouse

```xml
<!-- config.xml -->
<max_threads>8</max_threads>
<max_memory_usage>10000000000</max_memory_usage>
<max_bytes_before_external_group_by>5000000000</max_bytes_before_external_group_by>
```

## Cost Optimization

### AWS

- Use Spot Instances for Flink workers
- Enable S3 Intelligent-Tiering for backups
- Use Reserved Instances for stable workloads

### GCP

- Use Preemptible VMs for batch processing
- Enable committed use discounts
- Use Cloud Storage Nearline for cold data

### General

- Implement data retention policies
- Use compression (snappy, zstd)
- Right-size resources based on metrics
