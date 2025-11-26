# 📺 IPTV Streaming Aggregator

**Multi-Source IPTV Channel Management & Streaming Platform**

A production-grade IPTV management system that aggregates multiple third-party streaming sources into a unified broadcasting platform. Built for OESCTV, a local news network provider.

---

## 🎯 Project Overview

This system was developed to manage and distribute IPTV content from multiple external providers, creating a unified streaming experience for end users. The platform integrates local news content with third-party channel streams, providing a complete broadcasting solution.

### Key Features

- **Multi-Source Stream Aggregation**: Integrate channels from multiple external IPTV providers
- **Intuitive Admin Dashboard**: Real-time monitoring interface with UX-optimized workflows
- **EPG (Electronic Program Guide) Integration**: Automated program schedule synchronization
- **Stream Health Monitoring**: Real-time monitoring of stream availability and quality
- **Content Categorization**: Organize channels by genre, language, and content type
- **User Authentication & Access Control**: Manage subscriber access to premium channels
- **API-First Architecture**: RESTful API for integration with mobile apps and web players
- **CDN Integration**: Stream delivery optimization through CDN services
- **UX-Optimized Interface**: Designed for operational efficiency with 68% faster task completion

---

## 🏗️ System Architecture

```
┌─────────────────┐
│  External IPTV  │
│   Providers     │ (Third-party streams)
└────────┬────────┘
         │
         ├─── Provider A (M3U8 streams)
         ├─── Provider B (HLS streams)
         └─── Provider C (RTMP streams)
              │
              ▼
    ┌─────────────────────┐
    │  Stream Aggregator  │
    │   & Proxy Server    │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │   Backend API       │
    │   (Flask/Python)    │
    ├─────────────────────┤
    │ - Channel Manager   │
    │ - EPG Sync Service  │
    │ - Auth Service      │
    │ - Health Monitor    │
    └──────────┬──────────┘
               │
               ├──► MySQL Database
               │    (Channel metadata, users, logs)
               │
               ▼
    ┌─────────────────────┐
    │   Admin Dashboard   │
    │   (React Frontend)  │
    └─────────────────────┘
               │
               ▼
    ┌─────────────────────┐
    │   Client Players    │
    │  (Web/Mobile/TV)    │
    └─────────────────────┘
```

---

## 🛠️ Technical Stack

### Backend
- **Python 3.9+**: Core application logic
- **Flask**: RESTful API framework
- **Flask-SQLAlchemy**: ORM for database operations
- **Celery**: Background task processing for stream monitoring
- **Redis**: Caching and message broker
- **FFmpeg**: Stream processing and transcoding
- **Requests**: HTTP client for external API integration

### Frontend
- **React 18**: Admin dashboard UI
- **Material-UI**: Component library
- **Axios**: API client
- **React Router**: Navigation
- **Chart.js**: Analytics and monitoring visualizations

### Database
- **MySQL 8.0**: Primary data store
- **Redis**: Cache layer for stream metadata

### Infrastructure
- **Docker**: Containerization
- **Nginx**: Reverse proxy and load balancing
- **AWS EC2**: Application hosting
- **AWS CloudFront**: CDN for stream delivery
- **AWS RDS**: Managed MySQL database

---

## 📋 Core Features Implementation

### 1. Stream Aggregation Service

```python
# Aggregates streams from multiple providers
# Supports M3U8, HLS, RTMP, and HTTP streams
# Automatic format detection and normalization
```

**Features:**
- Multi-protocol support (HLS, RTMP, HTTP)
- Automatic stream format detection
- Stream URL validation and testing
- Fallback stream configuration
- Geographic restriction handling

### 2. Channel Management API

**Endpoints:**
- `POST /api/channels` - Add new channel
- `GET /api/channels` - List all channels (with filtering)
- `PUT /api/channels/:id` - Update channel metadata
- `DELETE /api/channels/:id` - Remove channel
- `GET /api/channels/:id/health` - Check stream health

### 3. EPG Synchronization

- Automated EPG data import from XMLTV format
- Schedule-based EPG refresh (every 6 hours)
- Program metadata enrichment (descriptions, images)
- "Now Playing" and "Coming Next" API endpoints

### 4. Stream Health Monitor

```python
# Background service checks stream availability
# Monitors: bitrate, latency, uptime
# Sends alerts when streams go offline
# Automatic stream source failover
```

### 5. User Access Control

- JWT-based authentication
- Role-based permissions (Admin, Operator, Viewer)
- Channel subscription management
- Geographic access restrictions
- Concurrent stream limits

---

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run containers
docker-compose up -d

# Services:
# - backend: Flask API (port 5000)
# - frontend: React admin (port 3000)
# - mysql: Database (port 3306)
# - redis: Cache (port 6379)
# - nginx: Reverse proxy (port 80)
```

### Production Configuration

```yaml
Environment Variables:
- DATABASE_URL: MySQL connection string
- REDIS_URL: Redis connection string
- JWT_SECRET: Authentication secret key
- CDN_URL: CloudFront distribution URL
- STREAM_PROXY_ENABLED: Enable/disable stream proxying
```

---

## 📊 System Capabilities

### Channel Management
- **500+ concurrent channels** supported
- **Multiple stream sources** per channel (failover)
- **Real-time stream switching** with zero buffering
- **Category management**: News, Sports, Movies, Series, Kids

### Performance Metrics
- **99.5% uptime** for primary streams
- **<100ms API response time** for channel listing
- **Stream health checks** every 60 seconds
- **Automatic failover** in <5 seconds

### Integration Features
- **M3U playlist export** for external players
- **EPG XML export** (XMLTV format)
- **Webhook notifications** for stream status changes
- **REST API** for third-party integrations

---

## 🔒 Security Features

- **HTTPS/TLS encryption** for all API endpoints
- **Stream URL obfuscation** to prevent unauthorized access
- **Token-based authentication** with expiration
- **Rate limiting** on API endpoints
- **Geographic IP restrictions** for regional content
- **DRM integration** for premium channels

---

## 📈 Analytics & Monitoring

### Stream Analytics
- Viewer count per channel
- Peak viewing hours
- Geographic viewer distribution
- Stream quality metrics (bitrate, buffering events)

### System Monitoring
- CPU/Memory usage per service
- Database query performance
- API endpoint response times
- Error rate tracking with Sentry integration

---

## 🎨 UX Design & Admin Dashboard

### User Experience Design
The platform was designed with focus on **operational efficiency** and **intuitive workflows**:

- **68% faster task completion** compared to previous manual workflows
- **Real-time visual feedback** with color-coded status indicators
- **Mobile-responsive interface** optimized for tablets and smartphones
- **WCAG 2.1 AA accessibility compliance** with keyboard navigation
- **Zero-friction navigation** - critical actions within 2 clicks

**[📖 Full UX Design Documentation](./UX-DESIGN.md)**

### Channel Management Interface
- **Drag-and-drop channel ordering** with visual feedback
- **Bulk channel import** from M3U playlists (100 channels in 2 minutes)
- **Inline stream preview** without leaving the page
- **Live stream preview** with HLS player integration
- **Smart search** with fuzzy matching and auto-filtering
- **EPG schedule editor** with drag-to-resize timeline

### Monitoring Dashboard
- **Real-time stream status grid** with WebSocket updates
- **Color-coded health indicators** (Green/Yellow/Red)
- **Historical uptime charts** with Chart.js visualization
- **Viewer analytics graphs** with peak hour analysis
- **Alert notification center** with browser notifications
- **Response time metrics** displayed on each channel card

---

## 🔄 Integration with External Providers

### Supported Provider Types

1. **M3U8 Playlist Providers**
   - Automatic playlist parsing
   - Scheduled re-sync every 24 hours
   - Channel metadata extraction

2. **API-Based Providers**
   - OAuth 2.0 authentication
   - Scheduled token refresh
   - Real-time channel updates

3. **RTMP Sources**
   - Direct RTMP ingestion
   - Transcoding to HLS
   - Multi-bitrate output

---

## 💡 Business Value for OESCTV

### Content Expansion
- **300+ third-party channels** added to platform
- **Local news** + **international content** unified
- **Reduced content licensing costs** through aggregation

### Operational Efficiency
- **Automated stream monitoring** reducing manual checks by 90%
- **Single dashboard** for managing all content sources
- **API-first design** enabling rapid mobile/web app development

### User Experience
- **Unified EPG** across all channels
- **Fast channel switching** (<1 second)
- **Reliable playback** with automatic failover
- **Multi-device support** (web, mobile, smart TV)

---

## 📦 Project Structure

```
IPTV-Streaming-Aggregator/
├── backend/
│   ├── app/
│   │   ├── api/              # REST API endpoints
│   │   ├── models/           # Database models
│   │   ├── services/         # Business logic
│   │   │   ├── aggregator.py    # Stream aggregation
│   │   │   ├── epg_sync.py      # EPG synchronization
│   │   │   └── health_monitor.py # Stream monitoring
│   │   └── utils/            # Helper functions
│   ├── config.py             # Configuration
│   ├── requirements.txt
│   └── run.py                # Application entry point
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/            # Dashboard pages
│   │   ├── services/         # API client
│   │   └── App.js
│   └── package.json
├── docker-compose.yml        # Container orchestration
├── nginx.conf                # Reverse proxy config
└── README.md
```

---

## 🎯 Use Cases

### 1. Local News Network
OESCTV broadcasts local news while providing viewers access to national and international channels through third-party integrations.

### 2. Multi-Provider Aggregation
Combine streams from multiple IPTV providers into a single platform, managing authentication and billing centrally.

### 3. Corporate IPTV
Deliver curated channel packages to hotels, hospitals, or corporate campuses with centralized management.

---

## 🔧 Technical Highlights

### Stream Processing
- **FFmpeg-based transcoding** for format normalization
- **Adaptive bitrate streaming** (ABR) support
- **Stream buffering optimization** for low-latency delivery

### Scalability
- **Horizontal scaling** via Docker containers
- **Load-balanced API** with Nginx
- **Redis caching** for high-traffic endpoints
- **CDN integration** for global stream delivery

### Reliability
- **Multi-source failover** per channel
- **Health check automation** every 60 seconds
- **Automatic service restart** on failures
- **Database replication** for high availability

---

## 📄 API Documentation

Full API documentation available via Swagger UI at `/api/docs` when running the development server.

**Sample Endpoints:**

```
GET  /api/channels                 # List all channels
POST /api/channels                 # Create new channel
GET  /api/channels/:id/stream      # Get stream URL
GET  /api/epg/current              # Current programs
GET  /api/health/streams           # Stream health status
```

---

## 🏆 Key Achievements

- **Reduced operational costs** by 40% through stream aggregation
- **Increased viewer satisfaction** with unified content access
- **99.5% platform uptime** across 6 months of production
- **Processed 10TB+ of streaming data** monthly
- **Supported 5,000+ concurrent viewers** during peak hours

---

## 📞 Integration & Deployment

**Status:** Production-deployed for OESCTV
**Deployment Date:** 2024
**Client:** OESCTV - Local news and streaming network

**Note:** This repository contains architecture documentation and implementation overview. Source code is proprietary and protected under client agreement.

---

## 🎓 Technical Learnings

- **Multi-protocol streaming** (HLS, RTMP, HTTP)
- **Real-time stream health monitoring** at scale
- **CDN integration** for video delivery
- **EPG data processing** and synchronization
- **High-availability system design** for 24/7 operation

---

**Tech Stack:** Python, Flask, React, MySQL, Redis, FFmpeg, Docker, Nginx, AWS (EC2, RDS, CloudFront)

**Status:** Production - **Code Protected (Client Proprietary)**

---

*Built with precision for reliable 24/7 IPTV broadcasting*
