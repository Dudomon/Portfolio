# Deployment Guide

## Prerequisites

### Required Tools
- kubectl 1.28+
- Helm 3.0+
- Docker 20.10+
- Minikube or cloud Kubernetes cluster

### System Requirements
- Minimum 8GB RAM
- 4 CPU cores
- 50GB disk space

## Local Development Deployment

### Step 1: Start Minikube

```bash
minikube start --cpus=4 --memory=8192 --disk-size=50g

# Enable required addons
minikube addons enable ingress
minikube addons enable metrics-server
```

### Step 2: Build Docker Images

```bash
# Build all service images
make build

# Or build individually
docker build -t order-service:latest ./services/order-service
docker build -t notification-service:latest ./services/notification-service
```

### Step 3: Load Images to Minikube

```bash
minikube image load order-service:latest
minikube image load notification-service:latest
```

### Step 4: Create Namespace

```bash
kubectl create namespace ecommerce
kubectl config set-context --current --namespace=ecommerce
```

### Step 5: Deploy Infrastructure

```bash
# Deploy databases and message broker
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/postgres-statefulset.yaml
kubectl apply -f k8s/rabbitmq-statefulset.yaml

# Wait for infrastructure to be ready
kubectl wait --for=condition=ready pod -l app=postgres --timeout=300s
kubectl wait --for=condition=ready pod -l app=rabbitmq --timeout=300s
```

### Step 6: Deploy Services

```bash
# Deploy microservices
kubectl apply -f k8s/order-service-deployment.yaml

# Verify deployment
kubectl get pods
kubectl get services
```

### Step 7: Deploy Ingress

```bash
kubectl apply -f k8s/ingress.yaml

# Get Minikube IP
minikube ip

# Add to /etc/hosts (or C:\Windows\System32\drivers\etc\hosts on Windows)
# <minikube-ip> ecommerce.local
```

### Step 8: Verify Deployment

```bash
# Check all pods are running
kubectl get pods

# Check services
kubectl get svc

# Test health endpoints
curl http://ecommerce.local/api/orders/health
```

## Production Deployment

### Cloud Provider Setup

#### AWS EKS

```bash
# Create EKS cluster
eksctl create cluster \
  --name ecommerce-prod \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 5

# Configure kubectl
aws eks update-kubeconfig --name ecommerce-prod --region us-east-1
```

#### GCP GKE

```bash
# Create GKE cluster
gcloud container clusters create ecommerce-prod \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-2 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 5

# Get credentials
gcloud container clusters get-credentials ecommerce-prod --zone us-central1-a
```

#### Azure AKS

```bash
# Create AKS cluster
az aks create \
  --resource-group ecommerce-rg \
  --name ecommerce-prod \
  --node-count 3 \
  --node-vm-size Standard_D2s_v3 \
  --enable-cluster-autoscaler \
  --min-count 2 \
  --max-count 5

# Get credentials
az aks get-credentials --resource-group ecommerce-rg --name ecommerce-prod
```

### Helm Deployment

```bash
# Add Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Create namespace
kubectl create namespace ecommerce-prod

# Deploy with Helm
helm install ecommerce ./helm/ecommerce \
  --namespace ecommerce-prod \
  --values ./helm/values-prod.yaml \
  --wait

# Verify deployment
helm list -n ecommerce-prod
kubectl get pods -n ecommerce-prod
```

### Production Configuration

Create `helm/values-prod.yaml`:

```yaml
global:
  namespace: ecommerce-prod
  imagePullPolicy: Always

orderService:
  replicaCount: 3
  resources:
    requests:
      memory: "512Mi"
      cpu: "500m"
    limits:
      memory: "1Gi"
      cpu: "1000m"
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20

postgres:
  persistence:
    enabled: true
    size: 100Gi
    storageClass: fast-ssd

rabbitmq:
  persistence:
    enabled: true
    size: 50Gi
    storageClass: fast-ssd

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  tls:
    - secretName: ecommerce-tls
      hosts:
        - api.ecommerce.com
```

## Database Initialization

### PostgreSQL

```bash
# Connect to PostgreSQL pod
kubectl exec -it postgres-0 -n ecommerce -- psql -U postgres

# Create databases
CREATE DATABASE orders;
CREATE DATABASE payments;

# Run migrations
kubectl exec -it <order-service-pod> -- alembic upgrade head
```

### MongoDB

```bash
# Connect to MongoDB pod
kubectl exec -it mongodb-0 -n ecommerce -- mongosh

# Create collections
use products;
db.createCollection("products");
db.createCollection("inventory");
```

## Monitoring Setup

### Prometheus

```bash
# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# Access Prometheus UI
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
```

### Grafana

```bash
# Get Grafana password
kubectl get secret -n monitoring prometheus-grafana \
  -o jsonpath="{.data.admin-password}" | base64 --decode

# Access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
```

### Jaeger

```bash
# Install Jaeger
kubectl create namespace observability
kubectl apply -f https://github.com/jaegertracing/jaeger-operator/releases/download/v1.50.0/jaeger-operator.yaml -n observability

# Deploy Jaeger instance
kubectl apply -f k8s/jaeger.yaml

# Access Jaeger UI
kubectl port-forward -n observability svc/jaeger-query 16686:16686
```

## SSL/TLS Configuration

### Install cert-manager

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f k8s/cluster-issuer.yaml
```

### Configure TLS Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ecommerce-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.ecommerce.com
    secretName: ecommerce-tls
  rules:
  - host: api.ecommerce.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: order-service
            port:
              number: 8002
```

## Backup and Disaster Recovery

### Database Backups

```bash
# PostgreSQL backup
kubectl exec postgres-0 -n ecommerce -- \
  pg_dump -U postgres orders > backup-orders-$(date +%Y%m%d).sql

# MongoDB backup
kubectl exec mongodb-0 -n ecommerce -- \
  mongodump --out=/backup/$(date +%Y%m%d)
```

### Automated Backups with CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15-alpine
            command:
            - /bin/sh
            - -c
            - pg_dump -h postgres -U postgres orders > /backup/backup-$(date +%Y%m%d).sql
            volumeMounts:
            - name: backup
              mountPath: /backup
          volumes:
          - name: backup
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

## Scaling

### Manual Scaling

```bash
# Scale deployment
kubectl scale deployment order-service --replicas=5 -n ecommerce

# Scale StatefulSet
kubectl scale statefulset postgres --replicas=3 -n ecommerce
```

### Auto-scaling

HPA is configured automatically. Monitor with:

```bash
kubectl get hpa -n ecommerce
kubectl describe hpa order-service-hpa -n ecommerce
```

## Rolling Updates

### Update Service Image

```bash
# Update image
kubectl set image deployment/order-service \
  order-service=order-service:v2.0.0 \
  -n ecommerce

# Monitor rollout
kubectl rollout status deployment/order-service -n ecommerce

# Rollback if needed
kubectl rollout undo deployment/order-service -n ecommerce
```

### Zero-Downtime Deployment

Ensure proper configuration:

```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  minReadySeconds: 10
```

## Health Checks

### Verify Service Health

```bash
# Check all pods
kubectl get pods -n ecommerce

# Check specific service
kubectl logs -f deployment/order-service -n ecommerce

# Execute health check
kubectl exec -it <pod-name> -n ecommerce -- curl localhost:8002/health
```

## Troubleshooting

### Pod Not Starting

```bash
# Describe pod
kubectl describe pod <pod-name> -n ecommerce

# Check logs
kubectl logs <pod-name> -n ecommerce

# Check events
kubectl get events -n ecommerce --sort-by='.lastTimestamp'
```

### Service Not Accessible

```bash
# Check service
kubectl get svc -n ecommerce
kubectl describe svc order-service -n ecommerce

# Check endpoints
kubectl get endpoints -n ecommerce

# Test from another pod
kubectl run test --image=curlimages/curl -it --rm -- \
  curl http://order-service:8002/health
```

### Database Connection Issues

```bash
# Check database pod
kubectl logs postgres-0 -n ecommerce

# Test connection
kubectl exec -it <service-pod> -n ecommerce -- \
  psql -h postgres -U postgres -d orders -c "SELECT 1"
```

## Cleanup

### Remove Deployment

```bash
# Using Helm
helm uninstall ecommerce -n ecommerce

# Using kubectl
kubectl delete namespace ecommerce

# Stop Minikube
minikube stop
minikube delete
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy to Kubernetes

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker images
      run: make build
    
    - name: Push to registry
      run: |
        docker tag order-service:latest ${{ secrets.REGISTRY }}/order-service:${{ github.sha }}
        docker push ${{ secrets.REGISTRY }}/order-service:${{ github.sha }}
    
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/order-service \
          order-service=${{ secrets.REGISTRY }}/order-service:${{ github.sha }} \
          -n ecommerce-prod
```

## Security Best Practices

1. Use secrets for sensitive data
2. Enable RBAC
3. Implement network policies
4. Regular security updates
5. Container image scanning
6. TLS for all communications
7. Audit logging enabled
