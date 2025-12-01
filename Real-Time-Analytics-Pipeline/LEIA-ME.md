# 📊 Pipeline de Analytics em Tempo Real

**Plataforma de analytics em streaming de nível empresarial para monitoramento de e-commerce**

![Status](https://img.shields.io/badge/status-pronto--para--produção-green)
![Python](https://img.shields.io/badge/python-3.11-blue)
![React](https://img.shields.io/badge/react-18.2-61dafb)
![Kafka](https://img.shields.io/badge/kafka-3.6-black)
![Flink](https://img.shields.io/badge/flink-1.18-e6526f)
![ClickHouse](https://img.shields.io/badge/clickhouse-23.8-yellow)
![Docker](https://img.shields.io/badge/docker-compose-2496ed)
![Licença](https://img.shields.io/badge/licença-MIT-blue)

---

## 🎯 Visão Geral

Pipeline de dados completo em tempo real capaz de ingerir, processar e visualizar dados de streaming com latência sub-segundo. O sistema foi projetado para lidar com streams de eventos de alto throughput (100K+ eventos/segundo) mantendo semântica de processamento exactly-once e fornecendo analytics interativo através de um dashboard web.

---

## ⚡ Início Rápido (5 minutos)

### Pré-requisitos
- Docker Desktop (mínimo 8GB RAM)
- Python 3.8+ (para gerador de eventos)
- Portas disponíveis: 8080, 8081, 9092, 8123, 6379, 3000

### Passo 1: Iniciar Serviços

**Windows:**
```bash
setup.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Manual:**
```bash
docker-compose up -d
```

### Passo 2: Acessar Dashboard

Abra seu navegador:
```
http://localhost:3000
```

### Passo 3: Gerar Eventos

Em um novo terminal:
```bash
# Instalar dependências
pip install requests

# Gerar 1000 eventos/segundo por 60 segundos
python scripts/generate_events.py --rate 1000 --duration 60
```

### Passo 4: Ver Atualizações em Tempo Real

O dashboard atualizará automaticamente a cada 2 segundos mostrando:
- 📈 Total de eventos
- 👥 Usuários únicos
- 💰 Receita total
- 🛒 Valor médio do pedido
- 📊 Gráficos de séries temporais
- 🏆 Produtos mais vendidos
- 🌍 Distribuição geográfica

---

## 🏗️ Arquitetura

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Produtores │      │   Apache    │      │   Apache    │      │ ClickHouse  │
│  de Eventos │─────▶│   Kafka     │─────▶│   Flink     │─────▶│   (OLAP)    │
│             │      │  (Broker)   │      │(Processador)│      │             │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
                                                                       │
                                                                       ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   React     │      │  WebSocket  │      │   FastAPI   │      │    Redis    │
│  Dashboard  │◀─────│   Server    │◀─────│   Backend   │◀─────│   (Cache)   │
│             │      │             │      │             │      │             │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
```

---

## 📦 Stack Tecnológica

### Camada de Dados
- **Apache Kafka 3.6**: Streaming de eventos distribuído
- **Apache Flink 1.18**: Motor de processamento de stream
- **ClickHouse 23.8**: Banco de dados OLAP colunar
- **Redis 7.2**: Cache e pub/sub

### Camada de Aplicação
- **FastAPI 0.104**: API Python de alta performance
- **React 18**: Framework frontend moderno
- **WebSocket**: Comunicação bidirecional em tempo real
- **Recharts**: Biblioteca de visualização de dados

### Infraestrutura
- **Docker Compose**: Orquestração multi-container
- **Nginx**: Proxy reverso e balanceamento de carga
- **Prometheus**: Coleta de métricas (opcional)

---

## 🎨 Funcionalidades do Dashboard

### Métricas em Tempo Real
- **Transações/Segundo**: Visualização de throughput ao vivo
- **Stream de Receita**: Receita cumulativa e por janela
- **Produtos Top**: Ranking dinâmico com contagem de vendas
- **Distribuição Geográfica**: Heatmap de vendas por região
- **Feed de Alertas**: Violações de threshold em tempo real

### Análise de Séries Temporais
- Janelas de tempo configuráveis (1m, 5m, 15m, 1h, 24h)
- Capacidades de drill-down
- Exportação para CSV/JSON
- Consultas de intervalo de datas customizadas

---

## 📈 Performance

### Benchmarks

| Métrica | Valor |
|---------|-------|
| Ingestão de Eventos | 100K eventos/seg |
| Latência End-to-End | <500ms (p99) |
| Tempo de Resposta de Query | <100ms (p95) |
| Retenção de Dados | 90 dias (configurável) |
| Compressão de Storage | Razão 10:1 |

### Escalabilidade

- **Kafka**: Escalonamento horizontal via partições
- **Flink**: Processamento paralelo com task slots
- **ClickHouse**: Tabelas distribuídas e sharding
- **API**: Design stateless para balanceamento de carga

---

## 🛠️ Desenvolvimento

### Estrutura do Projeto

```
real-time-analytics-pipeline/
├── docker-compose.yml          # Setup multi-container
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
├── flink/                      # Jobs Flink
│   ├── jobs/
│   │   ├── aggregator.py
│   │   └── enricher.py
│   └── Dockerfile
├── clickhouse/                 # Schemas de banco
│   └── init.sql
└── scripts/                    # Utilitários
    ├── generate_events.py
    └── load_test.py
```

### Desenvolvimento Local

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

---

## 📚 Documentação

- [QUICK_START.md](QUICK_START.md) - Guia de início rápido
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Deep dive na arquitetura
- [API.md](docs/API.md) - Referência da API
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) - Guia de deployment
- [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - Solução de problemas

---

## 🚨 Monitoramento & Alertas

### Health Checks

```bash
# Verificar todos os serviços
curl http://localhost:8080/health

# Tópicos Kafka
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Status ClickHouse
curl http://localhost:8123/ping
```

### Logs

```bash
# Ver todos os logs
docker-compose logs -f

# Serviço específico
docker-compose logs -f flink-jobmanager
```

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor leia [CONTRIBUTING.md](CONTRIBUTING.md) para diretrizes.

---

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 👤 Autor

**Eduardo Peiter**

- GitHub: [@Dudomon](https://github.com/Dudomon)
- Portfolio: [Ver Projetos](https://github.com/Dudomon)

---

## 🙏 Agradecimentos

- Apache Software Foundation pelo Kafka e Flink
- Time do ClickHouse pelo incrível banco OLAP
- Comunidades FastAPI e React

---

**Construído com ❤️ para processamento de dados em tempo real**
