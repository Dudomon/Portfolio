#!/bin/bash

# Local Development Setup Script

set -e

echo "Setting up E-Commerce Microservices Platform locally"
echo "====================================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."
command -v minikube >/dev/null 2>&1 || { echo "minikube is required but not installed. Aborting." >&2; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required but not installed. Aborting." >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "docker is required but not installed. Aborting." >&2; exit 1; }
echo "All prerequisites met!"
echo ""

# Start Minikube
echo "Starting Minikube..."
minikube start --cpus=4 --memory=8192 --disk-size=50g
echo ""

# Enable addons
echo "Enabling Minikube addons..."
minikube addons enable ingress
minikube addons enable metrics-server
echo ""

# Build Docker images
echo "Building Docker images..."
eval $(minikube docker-env)
docker build -t order-service:latest ./services/order-service
docker build -t notification-service:latest ./services/notification-service
echo ""

# Create namespace
echo "Creating Kubernetes namespace..."
kubectl create namespace ecommerce || true
kubectl config set-context --current --namespace=ecommerce
echo ""

# Deploy infrastructure
echo "Deploying infrastructure (PostgreSQL, RabbitMQ)..."
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/postgres-statefulset.yaml
kubectl apply -f k8s/rabbitmq-statefulset.yaml
echo ""

# Wait for infrastructure
echo "Waiting for infrastructure to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres --timeout=300s
kubectl wait --for=condition=ready pod -l app=rabbitmq --timeout=300s
echo ""

# Deploy services
echo "Deploying microservices..."
kubectl apply -f k8s/order-service-deployment.yaml
echo ""

# Wait for services
echo "Waiting for services to be ready..."
kubectl wait --for=condition=ready pod -l app=order-service --timeout=300s
echo ""

# Deploy ingress
echo "Deploying ingress..."
kubectl apply -f k8s/ingress.yaml
echo ""

# Get Minikube IP
MINIKUBE_IP=$(minikube ip)
echo ""
echo "====================================================="
echo "Setup Complete!"
echo ""
echo "Minikube IP: $MINIKUBE_IP"
echo ""
echo "Add this to your /etc/hosts file:"
echo "$MINIKUBE_IP ecommerce.local"
echo ""
echo "Access the API at: http://ecommerce.local/api/orders"
echo ""
echo "Useful commands:"
echo "  kubectl get pods                    - View all pods"
echo "  kubectl logs -f <pod-name>          - View logs"
echo "  kubectl port-forward svc/order-service 8002:8002  - Port forward"
echo "  minikube dashboard                  - Open Kubernetes dashboard"
echo "====================================================="
