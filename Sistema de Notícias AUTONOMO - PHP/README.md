# 📰 Autonomous News Aggregation System / Sistema Autônomo de Agregação de Notícias

[English](#english) | [Português](#português)

---

<a name="english"></a>
## 🇺🇸 English

**Advanced autonomous news aggregation and distribution system with intelligent scraping and automated content management**

Enterprise-grade news aggregation platform that automatically collects, processes, and distributes news from multiple sources with anti-duplicate systems, intelligent categorization, and real-time updates.

![System Overview](./screenshots/sistema-overview.png)

---

### 🎯 Key Capabilities

- **Autonomous Collection**: Automated multi-source news gathering (RSS + Custom Scrapers)
- **Intelligent Deduplication**: Advanced similarity detection and duplicate prevention
- **Smart Categorization**: Automatic classification (Local, Regional, National)
- **Dynamic Retention**: Source-based content lifecycle management
- **Real-time Updates**: Automated refresh cycles with cron integration
- **Content Processing**: Image extraction, encoding normalization, format standardization

---

### 🛠️ Technology Stack

#### Backend
- **PHP 7.4+** - Core system logic
- **MySQL/MariaDB** - News database
- **WordPress Integration** - Content management system
- **Custom XML/RSS Parsers** - Feed processing
- **XPath/DOMDocument** - Advanced web scraping

#### Automation
- **Cron Jobs** - Scheduled collection
- **WordPress Cron** - Fallback automation
- **Custom Schedulers** - Smart timing algorithms

#### Features
- **Anti-duplicate System** - Title matching + 80% similarity detection
- **Lazy-loading Support** - Dynamic image extraction
- **Multi-source Aggregation** - Unified content pipeline
- **Priority Management** - Smart content ranking

---

### 📊 System Architecture

```
┌─────────────────────────────────────────────┐
│         NEWS SOURCES (Multiple)             │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐   │
│  │ RSS  │  │ RSS  │  │Scraper│ │Scraper│   │
│  │  G1  │  │ NSC  │  │Cooper│  │  Etc  │   │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬────┘   │
└─────┼─────────┼─────────┼─────────┼─────────┘
      │         │         │         │
      └─────────┴─────────┴─────────┘
                    │
       ┌────────────▼────────────┐
       │   COLLECTION ENGINE     │
       │  • RSS Parser           │
       │  • Custom Scrapers      │
       │  • Image Extraction     │
       │  • Encoding Normalize   │
       └────────────┬────────────┘
                    │
       ┌────────────▼────────────┐
       │  ANTI-DUPLICATE SYSTEM  │
       │  • Title Exact Match    │
       │  • 80% Similarity Check │
       │  • Time-based Filter    │
       └────────────┬────────────┘
                    │
       ┌────────────▼────────────┐
       │    DATABASE ENGINE      │
       │  • Smart Categorization │
       │  • Priority Sorting     │
       │  • Retention Rules      │
       └────────────┬────────────┘
                    │
       ┌────────────▼────────────┐
       │   DISTRIBUTION API      │
       │  • JSON Endpoints       │
       │  • Widget Integration   │
       │  • Real-time Updates    │
       └─────────────────────────┘
```

---

### 🎨 Core Features

#### 1. Multi-Source Aggregation
- Processes **10+ news sources** simultaneously
- Supports RSS/Atom feeds and custom scrapers
- Handles different encoding standards
- Extracts images from various formats

#### 2. Intelligent Deduplication
- **Exact Title Matching**: Prevents immediate duplicates
- **Similarity Detection**: 80% threshold for near-duplicates
- **Time-based Filtering**: Configurable retention windows
- **Multi-category Support**: Independent tracking per category

#### 3. Smart Content Management
- **Local News**: 72-hour retention
- **Regional News**: 24-hour retention
- **National News**: 24-hour retention
- **Priority System**: Local > Regional > National

#### 4. Automated Workflows
- Scheduled collection every 3 hours
- Automatic cleanup of old content
- Image caching and optimization
- Error handling and retry logic

---

### 📈 Performance Metrics

- **185+ PHP modules** for specialized tasks
- **Sub-second** API response times
- **10 news limit** for optimal performance
- **Multi-tier caching** for efficiency
- **Failsafe mechanisms** for reliability

---

### 🔒 Enterprise Features

✅ **Reliability**
- Automated health checks
- Error logging and monitoring
- Graceful degradation
- Backup collection methods

✅ **Scalability**
- Modular architecture
- Easy source addition
- Configurable limits
- Database optimization

✅ **Maintainability**
- Comprehensive documentation
- Debug utilities
- Version tracking
- Changelog management

---

### 🌐 Integration Capabilities

#### API Endpoints
- News retrieval (JSON)
- Category filtering
- Search functionality
- Real-time updates

#### WordPress Integration
- Custom widgets
- Admin panels
- Automated publishing
- Media management

#### Third-party Services
- TTS (Text-to-Speech) integration
- Google Gemini AI integration
- Radio metadata sync
- Social media posting (Instagram/Stories)

---

### ⚠️ Note on Repository

**This is a PROPRIETARY showcase repository.**

This repository contains **only documentation and screenshots** to demonstrate the system's capabilities. The actual implementation, including:

- ✗ Source code (PHP files)
- ✗ Database schemas
- ✗ Scraper logic
- ✗ API implementations
- ✗ Configuration files

...is **NOT included** for intellectual property protection.

---

### 📜 License

This project is **proprietary software**. All rights reserved.

The code and implementation details are confidential and not available for public use or distribution.

---

<a name="português"></a>
## 🇧🇷 Português

**Sistema avançado autônomo de agregação e distribuição de notícias com scraping inteligente e gestão automatizada de conteúdo**

Plataforma de agregação de notícias nível empresarial que coleta, processa e distribui notícias automaticamente de múltiplas fontes com sistema anti-duplicata, categorização inteligente e atualizações em tempo real.

---

### 🎯 Capacidades Principais

- **Coleta Autônoma**: Agregação automática multi-fonte (RSS + Scrapers Customizados)
- **Deduplicação Inteligente**: Detecção avançada de similaridade e prevenção de duplicatas
- **Categorização Inteligente**: Classificação automática (Local, Regional, Nacional)
- **Retenção Dinâmica**: Gestão de ciclo de vida baseada em fonte
- **Atualizações em Tempo Real**: Ciclos de refresh automatizados com integração cron
- **Processamento de Conteúdo**: Extração de imagens, normalização de encoding, padronização de formato

---

### 🛠️ Stack Tecnológica

#### Backend
- **PHP 7.4+** - Lógica central do sistema
- **MySQL/MariaDB** - Banco de dados de notícias
- **Integração WordPress** - Sistema de gestão de conteúdo
- **Parsers XML/RSS Customizados** - Processamento de feeds
- **XPath/DOMDocument** - Web scraping avançado

#### Automação
- **Cron Jobs** - Coleta agendada
- **WordPress Cron** - Automação fallback
- **Schedulers Customizados** - Algoritmos de timing inteligente

#### Recursos
- **Sistema Anti-duplicata** - Matching de título + detecção de 80% similaridade
- **Suporte Lazy-loading** - Extração dinâmica de imagens
- **Agregação Multi-fonte** - Pipeline unificado de conteúdo
- **Gestão de Prioridades** - Ranking inteligente de conteúdo

---

### 📊 Arquitetura do Sistema

```
┌─────────────────────────────────────────────┐
│      FONTES DE NOTÍCIAS (Múltiplas)        │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐   │
│  │ RSS  │  │ RSS  │  │Scraper│ │Scraper│   │
│  │  G1  │  │ NSC  │  │Cooper│  │  Etc  │   │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬────┘   │
└─────┼─────────┼─────────┼─────────┼─────────┘
      │         │         │         │
      └─────────┴─────────┴─────────┘
                    │
       ┌────────────▼────────────┐
       │   ENGINE DE COLETA      │
       │  • Parser RSS           │
       │  • Scrapers Customizados│
       │  • Extração de Imagens  │
       │  • Normalização Encoding│
       └────────────┬────────────┘
                    │
       ┌────────────▼────────────┐
       │  SISTEMA ANTI-DUPLICATA │
       │  • Match Exato de Título│
       │  • Check 80% Similaridad│
       │  • Filtro Temporal      │
       └────────────┬────────────┘
                    │
       ┌────────────▼────────────┐
       │    ENGINE DE DATABASE   │
       │  • Categorização Smart  │
       │  • Ordenação Prioridade │
       │  • Regras de Retenção   │
       └────────────┬────────────┘
                    │
       ┌────────────▼────────────┐
       │   API DE DISTRIBUIÇÃO   │
       │  • Endpoints JSON       │
       │  • Integração Widget    │
       │  • Updates Tempo Real   │
       └─────────────────────────┘
```

---

### 🎨 Recursos Principais

#### 1. Agregação Multi-Fonte
- Processa **10+ fontes de notícias** simultaneamente
- Suporta feeds RSS/Atom e scrapers customizados
- Trata diferentes padrões de encoding
- Extrai imagens de vários formatos

#### 2. Deduplicação Inteligente
- **Matching Exato de Título**: Previne duplicatas imediatas
- **Detecção de Similaridade**: Threshold de 80% para quase-duplicatas
- **Filtragem Temporal**: Janelas de retenção configuráveis
- **Suporte Multi-categoria**: Tracking independente por categoria

#### 3. Gestão Inteligente de Conteúdo
- **Notícias Locais**: Retenção de 72 horas
- **Notícias Regionais**: Retenção de 24 horas
- **Notícias Nacionais**: Retenção de 24 horas
- **Sistema de Prioridades**: Local > Regional > Nacional

#### 4. Workflows Automatizados
- Coleta agendada a cada 3 horas
- Limpeza automática de conteúdo antigo
- Cache e otimização de imagens
- Tratamento de erros e lógica de retry

---

### 📈 Métricas de Performance

- **185+ módulos PHP** para tarefas especializadas
- Tempos de resposta API **sub-segundo**
- **Limite de 10 notícias** para performance otimizada
- **Caching multi-camada** para eficiência
- **Mecanismos failsafe** para confiabilidade

---

### 🔒 Recursos Empresariais

✅ **Confiabilidade**
- Health checks automatizados
- Logging e monitoramento de erros
- Degradação graciosa
- Métodos de coleta backup

✅ **Escalabilidade**
- Arquitetura modular
- Adição fácil de fontes
- Limites configuráveis
- Otimização de database

✅ **Manutenibilidade**
- Documentação abrangente
- Utilidades de debug
- Tracking de versão
- Gestão de changelog

---

### 🌐 Capacidades de Integração

#### Endpoints API
- Recuperação de notícias (JSON)
- Filtragem por categoria
- Funcionalidade de busca
- Atualizações em tempo real

#### Integração WordPress
- Widgets customizados
- Painéis admin
- Publicação automatizada
- Gestão de mídia

#### Serviços Terceiros
- Integração TTS (Text-to-Speech)
- Integração Google Gemini AI
- Sincronização metadata de rádio
- Posting em redes sociais (Instagram/Stories)

---

### ⚠️ Nota sobre o Repositório

**Este é um repositório proprietário de SHOWCASE.**

Este repositório contém **apenas documentação e screenshots** para demonstrar as capacidades do sistema. A implementação real, incluindo:

- ✗ Código fonte (arquivos PHP)
- ✗ Schemas de database
- ✗ Lógica dos scrapers
- ✗ Implementações de API
- ✗ Arquivos de configuração

...NÃO está incluída para proteção de propriedade intelectual.

---

### 📜 Licença

Este projeto é **software proprietário**. Todos os direitos reservados.

O código e detalhes de implementação são confidenciais e não disponíveis para uso ou distribuição pública.

---

**Version / Versão**: 1.0
**Development Period / Período de Desenvolvimento**: August - November 2024
**Status**: Production / Produção

---

> **Note:** This repository is for portfolio demonstration purposes only. The actual system is proprietary and not open-source.

> **Nota:** Este repositório é apenas para fins de demonstração de portfólio. O sistema real é proprietário e não é open-source.
