# SaaS RAG with Observability

[English](#english) | [Português](#português)

![Status](https://img.shields.io/badge/status-production--ready-green)
![Python](https://img.shields.io/badge/python-3.10-blue)
![Next.js](https://img.shields.io/badge/next.js-14-black)
![FastAPI](https://img.shields.io/badge/fastapi-0.111-009688)
![License](https://img.shields.io/badge/license-MIT-blue)

Enterprise-grade SaaS support platform with Retrieval-Augmented Generation (RAG), multi-tenant isolation, and comprehensive observability through OpenTelemetry.

---

<a name="english"></a>
## English

### Overview

This project implements a complete SaaS support platform featuring RAG capabilities with multi-tenant isolation and enterprise observability. The system demonstrates document ingestion, context-aware chat responses, and distributed tracing for production monitoring.

### Key Features

- **RAG Implementation**: Persistent vector store per tenant using ChromaDB with sentence-transformers embeddings
- **Multi-Tenant Isolation**: Complete data separation by tenant ID
- **LLM Integration**: OpenAI GPT-4o-mini with deterministic fallback
- **Observability**: OpenTelemetry tracing, Prometheus metrics, structured logging
- **Security**: API key authentication, rate limiting, tenant isolation
- **Production-Ready**: Docker containerization, health checks, error handling

### Architecture

**Backend (FastAPI)**
- ChromaDB for vector storage
- Sentence-transformers for embeddings
- OpenTelemetry SDK for tracing
- Prometheus metrics endpoint
- Optional OpenAI integration

**Frontend (Next.js 14)**
- App Router architecture
- Document ingestion interface
- Chat interface with source display
- Real-time API communication

**Observability Stack**
- OpenTelemetry Collector
- OTLP exporter for traces
- Prometheus metrics
- Ready for Grafana, Tempo, Jaeger

### Quick Start

#### Local Development

**Backend:**
```bash
cd "Saas Rag + Observabilidade/backend"
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd "Saas Rag + Observabilidade/frontend"
npm install
cp .env.example .env.local
npm run dev
```

Access: http://localhost:3000

#### Docker Compose

```bash
cd "Saas Rag + Observabilidade"
docker compose up --build
```

Services:
- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- OTLP Collector: http://localhost:4318

### API Reference

**POST /ingest**

Ingest documents into tenant knowledge base.

Request:
```json
{
  "tenant_id": "tenant-demo",
  "documents": [
    {
      "id": "doc-1",
      "text": "Support hours: 9am to 6pm.",
      "metadata": {"lang": "en"}
    }
  ]
}
```

Response:
```json
{
  "ingested": 1
}
```

**POST /chat**

Query knowledge base with RAG.

Request:
```json
{
  "tenant_id": "tenant-demo",
  "question": "What are the support hours?"
}
```

Response:
```json
{
  "answer": "Support hours are from 9am to 6pm.",
  "sources": [
    {
      "id": "doc-1",
      "score": 0.91,
      "text": "Support hours: 9am to 6pm.",
      "metadata": {"lang": "en"}
    }
  ]
}
```

**Headers:**
- `X-API-Key`: Required when API_KEYS is configured

**GET /metrics**

Prometheus metrics endpoint.

**GET /health**

Health check endpoint.

### Configuration

**Backend Environment Variables:**
```bash
APP_ENV=local
OPENAI_API_KEY=sk-xxx  # Optional
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PATH=data/chroma
OTLP_ENDPOINT=http://localhost:4318/v1/traces
OTLP_INSECURE=true
RAG_TOP_K=3
API_KEYS=test-key
RATE_LIMIT_PER_MIN=60
```

**Frontend Environment Variables:**
```bash
NEXT_PUBLIC_API_BASE=http://localhost:8000
NEXT_PUBLIC_API_KEY=demo-key
```

### Observability

**Tracing:**
- OpenTelemetry OTLP exporter
- FastAPI instrumentation
- Logging instrumentation
- Trace correlation

**Metrics:**
- Request rate
- Response time
- Error rate
- Custom business metrics

**Integration:**
- Grafana Tempo
- Jaeger
- Honeycomb
- SigNoz

### Production Recommendations

**Vector Store:**
- Consider managed services (Pinecone, Weaviate)
- Implement HNSW indexing for performance
- Cache embeddings

**LLM:**
- Fine-tune for domain
- Implement guardrails
- Add content filtering
- Monitor token usage

**Observability:**
- Structured logging with trace correlation
- Latency and recall dashboards
- Alert configuration
- Error tracking

**Security:**
- OAuth/JWT authentication
- Request rate limiting
- Input validation
- Secrets management

### Testing

```bash
cd backend
pytest
```

Tests cover:
- Document ingestion
- Chat functionality
- Authentication
- Rate limiting

---

<a name="português"></a>
## Português

### Visão Geral

Este projeto implementa uma plataforma SaaS completa de suporte com capacidades RAG, isolamento multi-tenant e observabilidade enterprise. O sistema demonstra ingestão de documentos, respostas de chat com contexto e tracing distribuído para monitoramento em produção.

### Funcionalidades Principais

- **Implementação RAG**: Armazenamento vetorial persistente por tenant usando ChromaDB com embeddings sentence-transformers
- **Isolamento Multi-Tenant**: Separação completa de dados por ID de tenant
- **Integração LLM**: OpenAI GPT-4o-mini com fallback determinístico
- **Observabilidade**: Tracing OpenTelemetry, métricas Prometheus, logging estruturado
- **Segurança**: Autenticação por API key, rate limiting, isolamento de tenant
- **Pronto para Produção**: Containerização Docker, health checks, tratamento de erros

### Arquitetura

**Backend (FastAPI)**
- ChromaDB para armazenamento vetorial
- Sentence-transformers para embeddings
- OpenTelemetry SDK para tracing
- Endpoint de métricas Prometheus
- Integração OpenAI opcional

**Frontend (Next.js 14)**
- Arquitetura App Router
- Interface de ingestão de documentos
- Interface de chat com exibição de fontes
- Comunicação API em tempo real

**Stack de Observabilidade**
- OpenTelemetry Collector
- Exportador OTLP para traces
- Métricas Prometheus
- Pronto para Grafana, Tempo, Jaeger

### Início Rápido

#### Desenvolvimento Local

**Backend:**
```bash
cd "Saas Rag + Observabilidade/backend"
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd "Saas Rag + Observabilidade/frontend"
npm install
copy .env.example .env.local
npm run dev
```

Acesso: http://localhost:3000

#### Docker Compose

```bash
cd "Saas Rag + Observabilidade"
docker compose up --build
```

Serviços:
- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- OTLP Collector: http://localhost:4318

### Configuração

**Variáveis de Ambiente Backend:**
```bash
APP_ENV=local
OPENAI_API_KEY=sk-xxx  # Opcional
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PATH=data/chroma
OTLP_ENDPOINT=http://localhost:4318/v1/traces
OTLP_INSECURE=true
RAG_TOP_K=3
API_KEYS=test-key
RATE_LIMIT_PER_MIN=60
```

**Variáveis de Ambiente Frontend:**
```bash
NEXT_PUBLIC_API_BASE=http://localhost:8000
NEXT_PUBLIC_API_KEY=demo-key
```

### Observabilidade

**Tracing:**
- Exportador OTLP OpenTelemetry
- Instrumentação FastAPI
- Instrumentação de logging
- Correlação de traces

**Métricas:**
- Taxa de requisições
- Tempo de resposta
- Taxa de erros
- Métricas de negócio customizadas

**Integração:**
- Grafana Tempo
- Jaeger
- Honeycomb
- SigNoz

### Recomendações para Produção

**Armazenamento Vetorial:**
- Considerar serviços gerenciados (Pinecone, Weaviate)
- Implementar indexação HNSW para performance
- Cachear embeddings

**LLM:**
- Fine-tune para domínio
- Implementar guardrails
- Adicionar filtragem de conteúdo
- Monitorar uso de tokens

**Observabilidade:**
- Logging estruturado com correlação de trace
- Dashboards de latência e recall
- Configuração de alertas
- Rastreamento de erros

**Segurança:**
- Autenticação OAuth/JWT
- Rate limiting de requisições
- Validação de entrada
- Gestão de secrets

### Testes

```bash
cd backend
pytest
```

Testes cobrem:
- Ingestão de documentos
- Funcionalidade de chat
- Autenticação
- Rate limiting

### Autor

Eduardo Peiter
- GitHub: [@Dudomon](https://github.com/Dudomon)

### Licença

Este projeto está licenciado sob a Licença MIT.
