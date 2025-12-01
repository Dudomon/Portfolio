# E-Commerce Microservices Platform

[English](#english) | [Português](#português)

![Status](https://img.shields.io/badge/status-production--ready-green)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Go](https://img.shields.io/badge/go-1.21-00ADD8)
![Kubernetes](https://img.shields.io/badge/kubernetes-1.28-326CE5)
![RabbitMQ](https://img.shields.io/badge/rabbitmq-3.12-FF6600)
![License](https://img.shields.io/badge/license-MIT-blue)

Enterprise-grade microservices architecture for e-commerce platform with event-driven communication and Kubernetes orchestration.

---

<a name="english"></a>
## English

### Overview

This project implements a complete e-commerce platform using microservices architecture with event-driven communication patterns. The system demonstrates distributed systems design, service orchestration with Kubernetes, asynchronous messaging with RabbitMQ, and enterprise observability practices.

The platform handles core e-commerce operations including product catalog management, shopping cart functionality, order processing with saga pattern for distributed transactions, payment processing, and real-time notifications.

### Architecture

The system consists of four core microservices communicating through an event bus:

**Product Service** (Go)
- Product catalog management
- Inventory tracking
- Search and filtering capabilities
- MongoDB for document storage

**Order Service** (Python/FastAPI)
- Order creation and management
- Event sourcing implementation
- CQRS pattern for read/write separation
- PostgreSQL for transactional data

**Payment Service** (Go)
- Payment processing simulation
- Transaction management
- Idempotency handling
- PostgreSQL for payment records

**Notification Service** (Python)
- Email and SMS notifications
- Event-driven message consumption
- Template management
- Async processing with Celery

### Key Features

- Microservices architecture with clear service boundaries
- Event-driven communication using RabbitMQ
- Saga pattern for distributed transactions
- CQRS and Event Sourcing in Order Service
- Kubernetes deployment with Helm charts
- Horizontal Pod Autoscaling (HPA)
- Service mesh ready (Istio compatible)
- Distributed tracing with Jaeger
- Metrics collection with Prometheus
- Centralized logging with ELK stack
- API Gateway with Kong
- Circuit breaker pattern implementation
- Health checks and readiness probes

### Technology Stack

**Microservices**
- Go 1.21 (Product Service, Payment Service)
- Python 3.11 with FastAPI (Order Service, Notification Service)
- gRPC for inter-service communication
- REST APIs for external clients

**Message Broker**
- RabbitMQ 3.12 for event bus
- Dead letter queues for failed messages
- Message persistence and acknowledgments

**Databases**
- PostgreSQL 15 (Order Service, Payment Service)
- MongoDB 7.0 (Product Service)
- Redis 7.2 (Caching, Session management)

**Kubernetes**
- Deployments and StatefulSets
- Services (ClusterIP, LoadBalancer)
- Ingress for external access
- ConfigMaps and Secrets
- Horizontal Pod Autoscaler
- Persistent Volumes

**Observability**
- Prometheus for metrics collection
- Grafana for visualization dashboards
- Jaeger for distributed tracing
- Elasticsearch, Logstash, Kibana for logs
- Custom metrics and alerts

**Infrastructure**
- Helm 3 for package management
- Kong API Gateway
- Nginx Ingress Controller
- Docker for containerization

### Quick Start

#### Prerequisites

- Kubernetes cluster (Minikube, Kind, or cloud provider)
- kubectl 1.28+
- Helm 3.0+
- Docker 20.10+

#### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/microservices-ecommerce-platform.git
cd microservices-ecommerce-platform

# Install dependencies with Helm
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Deploy infrastructure (RabbitMQ, PostgreSQL, MongoDB, Redis)
kubectl create namespace ecommerce
helm install -n ecommerce infrastructure ./helm/infrastructure

# Deploy microservices
helm install -n ecommerce ecommerce ./helm/ecommerce

# Verify deployment
kubectl get pods -n ecommerce
kubectl get services -n ecommerce

# Access API Gateway
kubectl port-forward -n ecommerce svc/kong-proxy 8000:8000
```

#### Local Development

```bash
# Start local Kubernetes cluster
minikube start --cpus=4 --memory=8192

# Enable required addons
minikube addons enable ingress
minikube addons enable metrics-server

# Deploy services
make deploy-local

# Access services
minikube service list -n ecommerce
```

### Service Communication

Services communicate through two patterns:

**Synchronous (gRPC)**
- Direct service-to-service calls
- Used for queries and immediate responses
- Circuit breaker protection

**Asynchronous (Events)**
- Event publishing to RabbitMQ
- Event consumption by interested services
- Eventual consistency model

### Event Flow Example

Order Creation Flow:
```
1. Client → API Gateway → Order Service (Create Order)
2. Order Service → Event Bus (OrderCreated event)
3. Event Bus → Payment Service (Process Payment)
4. Payment Service → Event Bus (PaymentProcessed event)
5. Event Bus → Inventory Service (Reserve Items)
6. Inventory Service → Event Bus (ItemsReserved event)
7. Event Bus → Notification Service (Send Confirmation)
8. Order Service → Event Bus (OrderConfirmed event)
```

Failure Handling (Saga Pattern):
```
If Payment Fails:
1. Payment Service → Event Bus (PaymentFailed event)
2. Event Bus → Order Service (Cancel Order)
3. Order Service → Event Bus (OrderCancelled event)
4. Event Bus → Notification Service (Send Cancellation)
```

### API Documentation

Each service exposes OpenAPI/Swagger documentation:

- Product Service: `http://localhost:8001/docs`
- Order Service: `http://localhost:8002/docs`
- Payment Service: `http://localhost:8003/docs`
- Notification Service: `http://localhost:8004/docs`

API Gateway endpoint: `http://localhost:8000`

### Monitoring and Observability

**Prometheus Metrics**
```bash
# Access Prometheus UI
kubectl port-forward -n monitoring svc/prometheus 9090:9090

# View metrics
open http://localhost:9090
```

**Grafana Dashboards**
```bash
# Access Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000

# Default credentials: admin/admin
open http://localhost:3000
```

**Jaeger Tracing**
```bash
# Access Jaeger UI
kubectl port-forward -n monitoring svc/jaeger-query 16686:16686

open http://localhost:16686
```

### Testing

```bash
# Unit tests
make test-unit

# Integration tests
make test-integration

# End-to-end tests
make test-e2e

# Load testing
make test-load
```

### Deployment

**Development**
```bash
make deploy-dev
```

**Staging**
```bash
make deploy-staging
```

**Production**
```bash
make deploy-prod
```

### Scaling

Services automatically scale based on CPU and memory usage:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: order-service
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: order-service
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

Manual scaling:
```bash
kubectl scale deployment order-service --replicas=5 -n ecommerce
```

### Security

- TLS encryption for all inter-service communication
- JWT authentication for API Gateway
- Secrets management with Kubernetes Secrets
- Network policies for service isolation
- RBAC for Kubernetes access control
- Container image scanning
- Regular security updates

### Documentation

- [Architecture](docs/ARCHITECTURE.md) - Detailed system design
- [API Reference](docs/API.md) - Complete API documentation
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment
- [Development Guide](docs/DEVELOPMENT.md) - Local development setup
- [Monitoring Guide](docs/MONITORING.md) - Observability setup
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues

### Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

### Author

Eduardo Peiter
- GitHub: [@Dudomon](https://github.com/Dudomon)

---

<a name="português"></a>
## Português

### Visão Geral

Este projeto implementa uma plataforma de e-commerce completa usando arquitetura de microservices com padrões de comunicação event-driven. O sistema demonstra design de sistemas distribuídos, orquestração de serviços com Kubernetes, mensageria assíncrona com RabbitMQ e práticas de observabilidade enterprise.

A plataforma gerencia operações core de e-commerce incluindo gestão de catálogo de produtos, funcionalidade de carrinho de compras, processamento de pedidos com saga pattern para transações distribuídas, processamento de pagamentos e notificações em tempo real.

### Arquitetura

O sistema consiste em quatro microservices core comunicando através de um event bus:

**Product Service** (Go)
- Gestão de catálogo de produtos
- Rastreamento de inventário
- Capacidades de busca e filtragem
- MongoDB para armazenamento de documentos

**Order Service** (Python/FastAPI)
- Criação e gestão de pedidos
- Implementação de event sourcing
- Pattern CQRS para separação read/write
- PostgreSQL para dados transacionais

**Payment Service** (Go)
- Simulação de processamento de pagamento
- Gestão de transações
- Tratamento de idempotência
- PostgreSQL para registros de pagamento

**Notification Service** (Python)
- Notificações por email e SMS
- Consumo de mensagens event-driven
- Gestão de templates
- Processamento assíncrono com Celery

### Funcionalidades Principais

- Arquitetura de microservices com limites de serviço claros
- Comunicação event-driven usando RabbitMQ
- Saga pattern para transações distribuídas
- CQRS e Event Sourcing no Order Service
- Deployment Kubernetes com Helm charts
- Horizontal Pod Autoscaling (HPA)
- Service mesh ready (compatível com Istio)
- Distributed tracing com Jaeger
- Coleta de métricas com Prometheus
- Logging centralizado com stack ELK
- API Gateway com Kong
- Implementação de circuit breaker pattern
- Health checks e readiness probes

### Stack Tecnológica

**Microservices**
- Go 1.21 (Product Service, Payment Service)
- Python 3.11 com FastAPI (Order Service, Notification Service)
- gRPC para comunicação inter-serviços
- REST APIs para clientes externos

**Message Broker**
- RabbitMQ 3.12 para event bus
- Dead letter queues para mensagens falhadas
- Persistência e acknowledgments de mensagens

**Databases**
- PostgreSQL 15 (Order Service, Payment Service)
- MongoDB 7.0 (Product Service)
- Redis 7.2 (Caching, Gestão de sessão)

**Kubernetes**
- Deployments e StatefulSets
- Services (ClusterIP, LoadBalancer)
- Ingress para acesso externo
- ConfigMaps e Secrets
- Horizontal Pod Autoscaler
- Persistent Volumes

**Observabilidade**
- Prometheus para coleta de métricas
- Grafana para dashboards de visualização
- Jaeger para distributed tracing
- Elasticsearch, Logstash, Kibana para logs
- Métricas e alertas customizados

**Infraestrutura**
- Helm 3 para gestão de pacotes
- Kong API Gateway
- Nginx Ingress Controller
- Docker para containerização

### Início Rápido

#### Pré-requisitos

- Cluster Kubernetes (Minikube, Kind, ou cloud provider)
- kubectl 1.28+
- Helm 3.0+
- Docker 20.10+

#### Instalação

```bash
# Clonar repositório
git clone https://github.com/yourusername/microservices-ecommerce-platform.git
cd microservices-ecommerce-platform

# Instalar dependências com Helm
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Deploy infraestrutura (RabbitMQ, PostgreSQL, MongoDB, Redis)
kubectl create namespace ecommerce
helm install -n ecommerce infrastructure ./helm/infrastructure

# Deploy microservices
helm install -n ecommerce ecommerce ./helm/ecommerce

# Verificar deployment
kubectl get pods -n ecommerce
kubectl get services -n ecommerce

# Acessar API Gateway
kubectl port-forward -n ecommerce svc/kong-proxy 8000:8000
```

### Comunicação entre Serviços

Serviços comunicam através de dois padrões:

**Síncrono (gRPC)**
- Chamadas diretas service-to-service
- Usado para queries e respostas imediatas
- Proteção com circuit breaker

**Assíncrono (Events)**
- Publicação de eventos no RabbitMQ
- Consumo de eventos por serviços interessados
- Modelo de consistência eventual

### Monitoramento e Observabilidade

**Métricas Prometheus**
```bash
# Acessar UI Prometheus
kubectl port-forward -n monitoring svc/prometheus 9090:9090

open http://localhost:9090
```

**Dashboards Grafana**
```bash
# Acessar Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000

# Credenciais padrão: admin/admin
open http://localhost:3000
```

**Tracing Jaeger**
```bash
# Acessar UI Jaeger
kubectl port-forward -n monitoring svc/jaeger-query 16686:16686

open http://localhost:16686
```

### Testes

```bash
# Testes unitários
make test-unit

# Testes de integração
make test-integration

# Testes end-to-end
make test-e2e

# Teste de carga
make test-load
```

### Segurança

- Criptografia TLS para toda comunicação inter-serviços
- Autenticação JWT para API Gateway
- Gestão de secrets com Kubernetes Secrets
- Network policies para isolamento de serviços
- RBAC para controle de acesso Kubernetes
- Scanning de imagens de container
- Atualizações regulares de segurança

### Documentação

- [Arquitetura](docs/ARCHITECTURE.md) - Design detalhado do sistema
- [Referência API](docs/API.md) - Documentação completa da API
- [Guia de Deployment](docs/DEPLOYMENT.md) - Deployment em produção
- [Guia de Desenvolvimento](docs/DEVELOPMENT.md) - Setup de desenvolvimento local
- [Guia de Monitoramento](docs/MONITORING.md) - Setup de observabilidade
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Problemas comuns

### Contribuindo

Contribuições são bem-vindas. Por favor leia [CONTRIBUTING.md](CONTRIBUTING.md) para diretrizes.

### Licença

Este projeto está licenciado sob a Licença MIT. Veja [LICENSE](LICENSE) para detalhes.

### Autor

Eduardo Peiter
- GitHub: [@Dudomon](https://github.com/Dudomon)
