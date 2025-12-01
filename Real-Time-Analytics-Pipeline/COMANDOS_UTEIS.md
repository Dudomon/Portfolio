# 🛠️ Comandos Úteis

Referência rápida de comandos para trabalhar com o projeto.

---

## 🚀 Iniciar/Parar Serviços

```bash
# Iniciar todos os serviços
docker-compose up -d

# Iniciar e ver logs
docker-compose up

# Parar todos os serviços
docker-compose down

# Parar e remover volumes (limpa dados)
docker-compose down -v

# Reiniciar todos os serviços
docker-compose restart

# Reiniciar serviço específico
docker-compose restart api
```

---

## 📊 Verificar Status

```bash
# Ver status de todos os serviços
docker-compose ps

# Ver logs de todos os serviços
docker-compose logs -f

# Ver logs de serviço específico
docker-compose logs -f api
docker-compose logs -f dashboard
docker-compose logs -f kafka
docker-compose logs -f flink-jobmanager

# Ver últimas 100 linhas
docker-compose logs --tail=100 api

# Ver uso de recursos
docker stats
```

---

## 🔍 Health Checks

```bash
# API health
curl http://localhost:8080/health

# ClickHouse ping
curl http://localhost:8123/ping

# Kafka topics
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Redis ping
docker exec redis redis-cli ping

# Flink dashboard
# Abrir http://localhost:8081
```

---

## 📥 Kafka

```bash
# Listar tópicos
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Criar tópico
docker exec kafka kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic test-topic \
  --partitions 3 \
  --replication-factor 1

# Descrever tópico
docker exec kafka kafka-topics --describe \
  --bootstrap-server localhost:9092 \
  --topic ecommerce-events

# Consumir mensagens
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic ecommerce-events \
  --from-beginning \
  --max-messages 10

# Produzir mensagem teste
docker exec -it kafka kafka-console-producer \
  --bootstrap-server localhost:9092 \
  --topic ecommerce-events
```

---

## 💾 ClickHouse

```bash
# Conectar ao cliente
docker exec -it clickhouse clickhouse-client \
  --user admin \
  --password admin123

# Executar query
docker exec clickhouse clickhouse-client \
  --user admin \
  --password admin123 \
  --query "SELECT count() FROM analytics.events"

# Ver databases
docker exec clickhouse clickhouse-client \
  --user admin \
  --password admin123 \
  --query "SHOW DATABASES"

# Ver tabelas
docker exec clickhouse clickhouse-client \
  --user admin \
  --password admin123 \
  --query "SHOW TABLES FROM analytics"

# Descrever tabela
docker exec clickhouse clickhouse-client \
  --user admin \
  --password admin123 \
  --query "DESCRIBE analytics.events"

# Ver últimos eventos
docker exec clickhouse clickhouse-client \
  --user admin \
  --password admin123 \
  --query "SELECT * FROM analytics.events ORDER BY timestamp DESC LIMIT 10"

# Contar eventos por tipo
docker exec clickhouse clickhouse-client \
  --user admin \
  --password admin123 \
  --query "SELECT event_type, count() FROM analytics.events GROUP BY event_type"
```

---

## 🔴 Redis

```bash
# Conectar ao cliente
docker exec -it redis redis-cli

# Ping
docker exec redis redis-cli ping

# Ver todas as chaves
docker exec redis redis-cli KEYS "*"

# Ver valor de chave
docker exec redis redis-cli GET "metrics:realtime"

# Limpar cache
docker exec redis redis-cli FLUSHALL

# Ver info
docker exec redis redis-cli INFO
```

---

## ⚙️ Flink

```bash
# Ver jobs rodando
curl http://localhost:8081/jobs

# Ver overview
curl http://localhost:8081/overview

# Acessar UI
# Abrir http://localhost:8081

# Ver logs do JobManager
docker-compose logs -f flink-jobmanager

# Ver logs do TaskManager
docker-compose logs -f flink-taskmanager

# Submeter job manualmente
docker exec flink-jobmanager flink run \
  -py /opt/flink/jobs/stream_processor.py
```

---

## 🔌 API

```bash
# Health check
curl http://localhost:8080/health

# Métricas em tempo real
curl http://localhost:8080/metrics/realtime

# Séries temporais
curl "http://localhost:8080/metrics/timeseries?window=1h"

# Top produtos
curl "http://localhost:8080/products/top?limit=5"

# Distribuição geográfica
curl http://localhost:8080/geo/distribution

# Alertas
curl "http://localhost:8080/alerts?limit=10"

# Enviar evento
curl -X POST http://localhost:8080/events \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "test_001",
    "event_type": "purchase",
    "user_id": "user_001",
    "product_id": "prod_001",
    "revenue": 99.99
  }'
```

---

## 🎨 Dashboard

```bash
# Acessar dashboard
# Abrir http://localhost:3000

# Ver logs
docker-compose logs -f dashboard

# Rebuild
docker-compose build dashboard
docker-compose up -d dashboard

# Entrar no container
docker exec -it dashboard sh
```

---

## 📊 Gerar Eventos

```bash
# Gerar 100 eventos/segundo por 60 segundos
python scripts/generate_events.py --rate 100 --duration 60

# Gerar 1000 eventos/segundo infinitamente
python scripts/generate_events.py --rate 1000

# Gerar com batch size customizado
python scripts/generate_events.py --rate 1000 --batch-size 50

# Gerar para API diferente
python scripts/generate_events.py --api-url http://localhost:8080 --rate 500
```

---

## 🧪 Teste de Carga

```bash
# Teste de carga básico
python scripts/load_test.py --rate 1000 --duration 60

# Teste de carga pesado
python scripts/load_test.py --rate 10000 --duration 300

# Teste com batch size customizado
python scripts/load_test.py --rate 5000 --duration 120 --batch-size 20
```

---

## 🐳 Docker

```bash
# Ver containers rodando
docker ps

# Ver todos os containers
docker ps -a

# Ver volumes
docker volume ls

# Ver networks
docker network ls

# Limpar tudo (cuidado!)
docker system prune -a

# Ver uso de espaço
docker system df

# Rebuild sem cache
docker-compose build --no-cache

# Rebuild serviço específico
docker-compose build --no-cache api
docker-compose up -d api
```

---

## 🔧 Troubleshooting

```bash
# Verificar portas em uso (Windows)
netstat -ano | findstr "8080 9092 8123 6379 3000"

# Verificar portas em uso (Linux/Mac)
lsof -i :8080,9092,8123,6379,3000

# Limpar e reiniciar tudo
docker-compose down -v
docker system prune -f
docker-compose up -d

# Ver logs de erro
docker-compose logs | grep -i error

# Verificar saúde de serviço
docker inspect --format='{{.State.Health.Status}}' api

# Entrar em container para debug
docker exec -it api bash
docker exec -it dashboard sh
docker exec -it clickhouse bash
```

---

## 📝 Git

```bash
# Ver status
git status

# Adicionar arquivos
git add Real-Time-Analytics-Pipeline/

# Commit
git commit -m "Add Real-Time Analytics Pipeline"

# Push
git push origin main

# Ver histórico
git log --oneline

# Ver diferenças
git diff
```

---

## 🎯 Atalhos com Makefile

```bash
# Iniciar
make start

# Parar
make stop

# Reiniciar
make restart

# Ver logs
make logs

# Limpar tudo
make clean

# Gerar eventos
make generate

# Ajuda
make help
```

---

## 💡 Dicas

### Ver métricas em tempo real
```bash
# Terminal 1: Logs da API
docker-compose logs -f api

# Terminal 2: Gerar eventos
python scripts/generate_events.py --rate 1000

# Terminal 3: Ver eventos no Kafka
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic ecommerce-events

# Browser: Dashboard
# http://localhost:3000
```

### Debug de performance
```bash
# Ver uso de CPU/Memória
docker stats

# Ver queries lentas no ClickHouse
docker exec clickhouse clickhouse-client \
  --user admin \
  --password admin123 \
  --query "SELECT query, query_duration_ms FROM system.query_log ORDER BY query_duration_ms DESC LIMIT 10"

# Ver lag do Kafka
docker exec kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --describe \
  --group flink-consumer-group
```

---

**Salve este arquivo para referência rápida! 📌**
