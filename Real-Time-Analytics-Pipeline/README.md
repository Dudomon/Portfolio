# Real-Time Analytics Pipeline

[English](#english) | [Português](#português)

![Status](https://img.shields.io/badge/status-production--ready-green)
![Python](https://img.shields.io/badge/python-3.11-blue)
![React](https://img.shields.io/badge/react-18.2-61dafb)
![Kafka](https://img.shields.io/badge/kafka-3.6-black)
![Flink](https://img.shields.io/badge/flink-1.18-e6526f)
![ClickHouse](https://img.shields.io/badge/clickhouse-23.8-yellow)
![Docker](https://img.shields.io/badge/docker-compose-2496ed)
![License](https://img.shields.io/badge/license-MIT-blue)

Enterprise-grade streaming analytics platform for e-commerce monitoring and real-time data processing.

---

<a name="english"></a>
## English

## Overview

This project implements a complete real-time data pipeline capable of ingesting, processing, and visualizing streaming data with sub-second latency. The system is designed to handle high-throughput event streams (100K+ events/second) while maintaining exactly-once processing semantics and providing interactive analytics through a web-based dashboard.

The architecture leverages industry-standard distributed systems including Apache Kafka for event streaming, Apache Flink for stateful stream processing, ClickHouse for OLAP analytics, and a React-based dashboard for real-time visualization.

## Key Features

- Real-time event ingestion with Apache Kafka (100K+ events/second throughput)
- Stateful stream processing with Apache Flink and exactly-once semantics
- Columnar OLAP storage with ClickHouse for fast analytical queries
- WebSocket-based live dashboard with React and Recharts
- Configurable alerting system with multiple severity levels
- Fault-tolerant architecture with automatic recovery and checkpointing
- Horizontally scalable design for production workloads
- Docker Compose orchestration for simplified deployment

## Architecture

```
Event Producers
    |
    v
Apache Kafka (Event Streaming)
    |
    v
Apache Flink (Stream Processing)
    |-- Windowed Aggregation
    |-- Stream Enrichment
    |-- Alert Evaluation
    |
    v
ClickHouse (OLAP Storage) <--> Redis (Cache)
    |
    v
FastAPI Backend (REST + WebSocket)
    |
    v
React Dashboard (Real-time UI)
```

## Quick Start

### Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Minimum 8GB RAM available for containers
- Available ports: 8080, 8081, 9092, 8123, 6379, 3000

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/real-time-analytics-pipeline.git
cd real-time-analytics-pipeline

# Start all services
docker-compose up -d

# Verify all services are healthy
docker-compose ps

# View logs
docker-compose logs -f

# Access the dashboard
http://localhost:3000
```

### Generate Sample Data

```bash
# Install Python dependencies
pip install requests

# Start event producer (simulates e-commerce transactions)
python scripts/generate_events.py --rate 1000 --duration 300

# Options:
#   --rate: Events per second (default: 100)
#   --duration: Duration in seconds (0 = infinite)
#   --batch-size: Batch size for API calls (default: 10)
```

## Tech Stack

### Data Layer
- Apache Kafka 3.6 - Distributed event streaming platform
- Apache Flink 1.18 - Stateful stream processing engine
- ClickHouse 23.8 - Columnar OLAP database
- Redis 7.2 - In-memory cache and pub/sub

### Application Layer
- FastAPI 0.104 - High-performance async Python API
- React 18 - Modern frontend framework with hooks
- WebSocket - Real-time bidirectional communication
- Recharts 2.10 - Composable charting library

### Infrastructure
- Docker Compose - Multi-container orchestration
- Python 3.11 - Backend runtime
- Node.js 18 - Frontend build tooling

## Use Cases

### E-Commerce Monitoring
- Real-time sales and revenue tracking
- Inventory level alerts and notifications
- Customer behavior pattern analysis
- Transaction anomaly detection

### IoT Sensor Data
- Device health and status monitoring
- Real-time anomaly detection
- Predictive maintenance scheduling
- Environmental metrics tracking

### Application Metrics
- API performance and latency monitoring
- Error rate tracking and alerting
- User activity and engagement analytics
- System resource utilization

## Dashboard Features

### Real-Time Metrics
- Live transaction throughput visualization
- Cumulative and windowed revenue tracking
- Dynamic product ranking by sales volume
- Geographic distribution with country-level breakdown
- Real-time alert feed with severity levels

### Time-Series Analysis
- Configurable time windows: 1m, 5m, 15m, 1h, 24h
- Interactive charts with zoom and pan
- Multiple metric overlays (revenue, events, users)
- Historical data queries with custom date ranges

## Configuration

### Environment Variables

```bash
# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_TOPIC=ecommerce-events

# ClickHouse Configuration
CLICKHOUSE_HOST=clickhouse
CLICKHOUSE_PORT=8123
CLICKHOUSE_DB=analytics

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
CORS_ORIGINS=http://localhost:3000
```

### Alert Thresholds

Edit `config/alerts.yaml`:

```yaml
alerts:
  - name: high_transaction_rate
    metric: transactions_per_second
    threshold: 5000
    operator: greater_than
    window: 60s
    
  - name: low_revenue
    metric: revenue_per_minute
    threshold: 1000
    operator: less_than
    window: 300s
```

## Performance

### Benchmarks

| Metric | Value |
|--------|-------|
| Event Ingestion | 100,000 events/sec |
| End-to-End Latency | <500ms (p99) |
| Query Response Time | <100ms (p95) |
| Data Retention | 90 days (configurable) |
| Storage Compression | 10:1 ratio |

### Scalability

**Kafka**: Horizontal scaling through topic partitioning and consumer groups

**Flink**: Parallel processing with configurable task slots and parallelism

**ClickHouse**: Distributed tables with sharding and replication

**API**: Stateless design enables unlimited horizontal scaling

## Testing

```bash
# Run unit tests
pytest tests/unit -v

# Run integration tests
pytest tests/integration -v

# Load testing
python scripts/load_test.py --duration 300 --rate 10000

# Code quality
flake8 api/ flink/
black api/ flink/ --check
```

## Documentation

- [Architecture Deep Dive](docs/ARCHITECTURE.md) - Detailed system design and component interactions
- [API Reference](docs/API.md) - Complete REST API documentation
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## Development

### Project Structure

```
real-time-analytics-pipeline/
├── docker-compose.yml          # Multi-container setup
├── kafka/                      # Kafka configuration
│   └── topics.sh
├── flink/                      # Flink jobs
│   ├── jobs/
│   │   ├── aggregator.py      # Windowed aggregations
│   │   └── enricher.py        # Stream enrichment
│   └── Dockerfile
├── clickhouse/                 # Database schemas
│   └── init.sql
├── api/                        # FastAPI backend
│   ├── main.py
│   ├── routes/
│   ├── models/
│   └── websocket.py
├── dashboard/                  # React frontend
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── services/
│   └── package.json
├── scripts/                    # Utilities
│   ├── generate_events.py     # Event producer
│   └── load_test.py
└── tests/
```

### Local Development

```bash
# Backend
cd api
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd dashboard
npm install
npm start
```

## Monitoring and Health Checks

### Service Health

```bash
# Check all services
curl http://localhost:8080/health

# Kafka topics
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

# ClickHouse status
curl http://localhost:8123/ping
```

### Logs

```bash
# View all logs
docker-compose logs -f

# Specific service
docker-compose logs -f flink-jobmanager
```

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

Eduardo Peiter
- GitHub: [@Dudomon](https://github.com/Dudomon)

## Acknowledgments

- Apache Software Foundation for Kafka and Flink
- ClickHouse development team
- FastAPI and React communities

---

<a name="português"></a>
## Português

### Visão Geral

Este projeto implementa um pipeline de dados completo em tempo real capaz de ingerir, processar e visualizar dados de streaming com latência sub-segundo. O sistema foi projetado para lidar com streams de eventos de alto throughput (mais de 100.000 eventos por segundo) mantendo semântica de processamento exactly-once e fornecendo analytics interativo através de um dashboard web.

A arquitetura utiliza sistemas distribuídos padrão da indústria incluindo Apache Kafka para streaming de eventos, Apache Flink para processamento stateful de streams, ClickHouse para analytics OLAP, e um dashboard baseado em React para visualização em tempo real.

### Funcionalidades Principais

- Ingestão de eventos em tempo real com Apache Kafka (throughput de mais de 100.000 eventos por segundo)
- Processamento stateful de streams com Apache Flink e semântica exactly-once
- Armazenamento OLAP colunar com ClickHouse para queries analíticas rápidas
- Dashboard ao vivo baseado em WebSocket com React e Recharts
- Sistema de alertas configurável com múltiplos níveis de severidade
- Arquitetura fault-tolerant com recuperação automática e checkpointing
- Design horizontalmente escalável para cargas de produção
- Orquestração Docker Compose para deployment simplificado

### Arquitetura

```
Produtores de Eventos
    |
    v
Apache Kafka (Streaming de Eventos)
    |
    v
Apache Flink (Processamento de Stream)
    |-- Agregação em Janelas
    |-- Enriquecimento de Stream
    |-- Avaliação de Alertas
    |
    v
ClickHouse (Armazenamento OLAP) <--> Redis (Cache)
    |
    v
Backend FastAPI (REST + WebSocket)
    |
    v
Dashboard React (UI em Tempo Real)
```

### Início Rápido

#### Pré-requisitos

- Docker 20.10+ e Docker Compose 2.0+
- Mínimo de 8GB RAM disponível para containers
- Portas disponíveis: 8080, 8081, 9092, 8123, 6379, 3000

#### Instalação

```bash
# Clonar o repositório
git clone https://github.com/yourusername/real-time-analytics-pipeline.git
cd real-time-analytics-pipeline

# Iniciar todos os serviços
docker-compose up -d

# Verificar que todos os serviços estão saudáveis
docker-compose ps

# Ver logs
docker-compose logs -f

# Acessar o dashboard
http://localhost:3000
```

#### Gerar Dados de Exemplo

```bash
# Instalar dependências Python
pip install requests

# Iniciar produtor de eventos (simula transações de e-commerce)
python scripts/generate_events.py --rate 1000 --duration 300

# Opções:
#   --rate: Eventos por segundo (padrão: 100)
#   --duration: Duração em segundos (0 = infinito)
#   --batch-size: Tamanho do batch para chamadas API (padrão: 10)
```

### Stack Tecnológica

#### Camada de Dados
- Apache Kafka 3.6: Plataforma de streaming de eventos distribuída
- Apache Flink 1.18: Motor de processamento stateful de streams
- ClickHouse 23.8: Banco de dados OLAP colunar
- Redis 7.2: Cache em memória e pub/sub

#### Camada de Aplicação
- FastAPI 0.104: API Python assíncrona de alta performance
- React 18: Framework frontend moderno com hooks
- WebSocket: Comunicação bidirecional em tempo real
- Recharts 2.10: Biblioteca de gráficos composável

#### Infraestrutura
- Docker Compose: Orquestração multi-container
- Python 3.11: Runtime backend
- Node.js 18: Ferramentas de build frontend

### Casos de Uso

#### Monitoramento de E-Commerce
- Rastreamento de vendas e receita em tempo real
- Alertas e notificações de níveis de inventário
- Análise de padrões de comportamento do cliente
- Detecção de anomalias em transações

#### Dados de Sensores IoT
- Monitoramento de saúde e status de dispositivos
- Detecção de anomalias em tempo real
- Agendamento de manutenção preditiva
- Rastreamento de métricas ambientais

#### Métricas de Aplicação
- Monitoramento de performance e latência de API
- Rastreamento e alertas de taxa de erro
- Analytics de atividade e engajamento de usuários
- Utilização de recursos do sistema

### Funcionalidades do Dashboard

#### Métricas em Tempo Real
- Visualização de throughput de transações ao vivo
- Rastreamento de receita cumulativa e por janela
- Ranking dinâmico de produtos por volume de vendas
- Distribuição geográfica com breakdown por país
- Feed de alertas em tempo real com níveis de severidade

#### Análise de Séries Temporais
- Janelas de tempo configuráveis: 1m, 5m, 15m, 1h, 24h
- Gráficos interativos com zoom e pan
- Sobreposição de múltiplas métricas (receita, eventos, usuários)
- Queries de dados históricos com intervalos de datas customizados

### Configuração

#### Variáveis de Ambiente

```bash
# Configuração Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_TOPIC=ecommerce-events

# Configuração ClickHouse
CLICKHOUSE_HOST=clickhouse
CLICKHOUSE_PORT=8123
CLICKHOUSE_DB=analytics

# Configuração Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Configuração API
API_HOST=0.0.0.0
API_PORT=8080
CORS_ORIGINS=http://localhost:3000
```

### Performance

#### Benchmarks

| Métrica | Valor |
|---------|-------|
| Ingestão de Eventos | 100.000 eventos/seg |
| Latência End-to-End | <500ms (p99) |
| Tempo de Resposta de Query | <100ms (p95) |
| Retenção de Dados | 90 dias (configurável) |
| Compressão de Storage | Razão 10:1 |

#### Escalabilidade

**Kafka**: Escalonamento horizontal através de particionamento de tópicos e grupos de consumidores

**Flink**: Processamento paralelo com task slots e paralelismo configuráveis

**ClickHouse**: Tabelas distribuídas com sharding e replicação

**API**: Design stateless permite escalonamento horizontal ilimitado

### Testes

```bash
# Executar testes unitários
pytest tests/unit -v

# Executar testes de integração
pytest tests/integration -v

# Teste de carga
python scripts/load_test.py --duration 300 --rate 10000

# Qualidade de código
flake8 api/ flink/
black api/ flink/ --check
```

### Documentação

- [Arquitetura Detalhada](docs/ARCHITECTURE.md): Design do sistema e interações entre componentes
- [Referência da API](docs/API.md): Documentação completa da REST API
- [Guia de Deployment](docs/DEPLOYMENT.md): Instruções de deployment em produção
- [Troubleshooting](docs/TROUBLESHOOTING.md): Problemas comuns e soluções

### Desenvolvimento

#### Estrutura do Projeto

```
real-time-analytics-pipeline/
├── docker-compose.yml          # Setup multi-container
├── kafka/                      # Configuração Kafka
│   └── topics.sh
├── flink/                      # Jobs Flink
│   ├── jobs/
│   │   ├── aggregator.py      # Agregações em janelas
│   │   └── enricher.py        # Enriquecimento de stream
│   └── Dockerfile
├── clickhouse/                 # Schemas de banco
│   └── init.sql
├── api/                        # Backend FastAPI
│   ├── main.py
│   ├── routes/
│   ├── models/
│   └── websocket.py
├── dashboard/                  # Frontend React
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── services/
│   └── package.json
├── scripts/                    # Utilitários
│   ├── generate_events.py     # Produtor de eventos
│   └── load_test.py
└── tests/
```

#### Desenvolvimento Local

```bash
# Backend
cd api
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd dashboard
npm install
npm start
```

### Monitoramento e Health Checks

#### Saúde dos Serviços

```bash
# Verificar todos os serviços
curl http://localhost:8080/health

# Tópicos Kafka
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Status ClickHouse
curl http://localhost:8123/ping
```

#### Logs

```bash
# Ver todos os logs
docker-compose logs -f

# Serviço específico
docker-compose logs -f flink-jobmanager
```

### Contribuindo

Contribuições são bem-vindas. Por favor leia [CONTRIBUTING.md](CONTRIBUTING.md) para diretrizes sobre como contribuir para este projeto.

### Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

### Autor

Eduardo Peiter
- GitHub: [@Dudomon](https://github.com/Dudomon)

### Agradecimentos

- Apache Software Foundation pelo Kafka e Flink
- Time de desenvolvimento do ClickHouse
- Comunidades FastAPI e React
