# Legal Document Analyzer with AWS Bedrock

Production-grade legal document analysis platform using AWS Bedrock (Claude 3.5 Sonnet) for Brazilian public sector applications. Built with PHP 8.2, featuring high-volume processing, comprehensive observability, and enterprise-grade security.

## Key Features

- AWS Bedrock integration with Claude 3.5 Sonnet for document analysis
- RESTful API with JWT authentication and role-based access control
- Asynchronous processing with SQS for high-volume workloads
- Redis caching for cost optimization and performance
- Comprehensive observability: CloudWatch Logs, Metrics, and X-Ray tracing
- S3 storage with KMS encryption for sensitive documents
- Rate limiting and security best practices
- Terraform infrastructure as code
- CI/CD pipeline with GitHub Actions

## Architecture

### High-Level Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTPS
       ▼
┌─────────────────────────────────────────────┐
│           API Gateway / Load Balancer        │
└─────────────────┬───────────────────────────┘
                  │
       ┌──────────┴──────────┐
       │                     │
       ▼                     ▼
┌─────────────┐      ┌─────────────┐
│  PHP API    │◄────►│   Redis     │
│  (Bedrock)  │      │   Cache     │
└─────┬───┬───┘      └─────────────┘
      │   │
      │   └──────────┐
      ▼              ▼
┌─────────────┐  ┌─────────────┐
│  S3 Bucket  │  │  SQS Queue  │
│  (KMS enc)  │  │  (Async)    │
└─────────────┘  └─────────────┘
      │              │
      └──────┬───────┘
             ▼
      ┌─────────────┐
      │  Bedrock    │
      │  (Claude)   │
      └─────────────┘
             │
             ▼
      ┌─────────────┐
      │ CloudWatch  │
      │  X-Ray      │
      └─────────────┘
```

### Component Responsibilities

**API Layer**
- Request routing and validation
- Authentication and authorization
- Rate limiting
- Error handling

**Services Layer**
- BedrockService: LLM inference with cost tracking
- S3Service: Document storage with encryption
- SQSService: Asynchronous job processing
- CacheService: Response caching for cost optimization
- ObservabilityService: Logging, metrics, and tracing

**Infrastructure**
- S3: Encrypted document storage
- SQS: Async processing queue with DLQ
- KMS: Encryption key management
- CloudWatch: Centralized logging and metrics
- X-Ray: Distributed tracing

## API Endpoints

### Authentication
```
POST /api/v1/auth/login
POST /api/v1/auth/refresh
```

### Documents
```
POST /api/v1/documents/upload       - Upload document
GET  /api/v1/documents/{id}         - Get document details
GET  /api/v1/documents              - List user documents
DELETE /api/v1/documents/{id}       - Delete document
```

### Analysis
```
POST /api/v1/analysis/extract       - Extract entities
POST /api/v1/analysis/classify      - Classify document type
POST /api/v1/analysis/summarize     - Generate summary
POST /api/v1/analysis/compare       - Compare two documents
GET  /api/v1/analysis/{id}          - Get analysis result
GET  /api/v1/analysis/{id}/status   - Check processing status
```

### Health
```
GET /health                          - Basic health check
GET /health/deep                     - Deep health check (all services)
```

## Installation

### Prerequisites
- PHP 8.2+
- Composer 2.x
- Redis 7.x
- AWS Account with Bedrock access

### Setup

1. Clone and install dependencies:
```bash
composer install
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your AWS credentials and configuration
```

3. Deploy infrastructure:
```bash
cd terraform
terraform init
terraform plan
terraform apply
```

4. Start development server:
```bash
php -S localhost:8000 -t public/
```

## Configuration

### Environment Variables

**AWS Configuration**
```
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
```

**Application Settings**
```
APP_ENV=production
APP_DEBUG=false
JWT_SECRET=your-jwt-secret-key
```

**Infrastructure**
```
S3_BUCKET_DOCUMENTS=legal-docs-storage
SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/.../legal-doc-processing
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
```

## Usage Examples

### Authentication
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo123"}'
```

### Upload Document
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "document=@contract.pdf" \
  -F "document_type=contrato"
```

### Extract Entities
```bash
curl -X POST http://localhost:8000/api/v1/analysis/extract \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Document text here...",
    "document_type": "sentenca"
  }'
```

### Classify Document
```bash
curl -X POST http://localhost:8000/api/v1/analysis/classify \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "Document text here..."}'
```

## Security Features

### Authentication & Authorization
- JWT-based authentication with configurable expiration
- Role-based access control (RBAC)
- Least privilege principle applied to all IAM policies

### Data Protection
- All documents encrypted at rest using AWS KMS
- TLS 1.2+ for data in transit
- S3 bucket with public access blocked
- Presigned URLs with time-limited access

### Application Security
- Input validation and sanitization
- Rate limiting to prevent abuse
- Secure error handling (no sensitive data in responses)
- Request ID tracking for auditability

## Observability

### Logging
- Structured JSON logs with context
- CloudWatch Logs integration
- Request/response correlation via request IDs
- Error tracking with stack traces (in debug mode)

### Metrics
Custom CloudWatch metrics tracked:
- `bedrock.request.success` - Successful Bedrock calls
- `bedrock.request.error` - Failed Bedrock calls
- `bedrock.latency` - Bedrock API latency (ms)
- `bedrock.cost` - Cost per request (USD)
- `bedrock.tokens.input` - Input tokens used
- `bedrock.tokens.output` - Output tokens used
- `api.request.count` - Total API requests
- `api.request.duration` - API latency (ms)
- `rate_limit.exceeded` - Rate limit violations
- `s3.upload.success` - Successful uploads
- `sqs.message.sent` - Messages queued

### Tracing
- AWS X-Ray integration for distributed tracing
- Trace IDs propagated across services
- Performance bottleneck identification

### Dashboards
CloudWatch dashboard includes:
- API request volume and latency
- Bedrock success/error rates
- Cost tracking per hour/day
- SQS queue depth and processing time

## Performance Optimization

### Caching Strategy
- Redis caching for identical requests
- SHA-256 hash-based cache keys
- Configurable TTL (default 1 hour)
- Cache hit/miss metrics tracked

### Cost Optimization
- Response caching reduces redundant Bedrock calls
- Batch processing for large documents
- Async processing for non-urgent requests
- Cost tracking per request and aggregated

### Scalability
- Stateless API design for horizontal scaling
- SQS for decoupled async processing
- Redis for distributed caching
- S3 for unlimited storage capacity

## Testing

### Run Tests
```bash
composer run-script test
```

### Code Quality
```bash
composer run-script cs      # PHP CodeSniffer
composer run-script stan    # PHPStan static analysis
composer run-script check   # Run all checks
```

## CI/CD Pipeline

GitHub Actions workflow includes:
1. Code quality checks (PSR-12, PHPStan)
2. Unit and integration tests
3. Security vulnerability scanning
4. Terraform validation
5. Infrastructure deployment
6. Application deployment
7. Post-deployment verification

## Infrastructure Management

### Terraform Resources
- S3 bucket with versioning and encryption
- KMS key for document encryption
- SQS queue with dead letter queue
- CloudWatch log groups and alarms
- IAM roles with least privilege policies
- CloudWatch dashboard

### Deploy Infrastructure
```bash
cd terraform
terraform apply
```

### View Outputs
```bash
terraform output
```

## Cost Estimation

**AWS Bedrock (Claude 3.5 Sonnet)**
- Input: $3.00 per million tokens
- Output: $15.00 per million tokens

**Example Analysis**
- 10,000 document analyses/month
- 2,000 tokens input + 500 tokens output per document
- Monthly cost: ~$135 USD

**Additional AWS Costs**
- S3: ~$0.023/GB/month
- SQS: First 1M requests free
- CloudWatch: First 5GB logs free
- KMS: $1/key/month

## Monitoring & Alerts

### CloudWatch Alarms
- High Bedrock error rate (> 10 errors in 5 min)
- High API latency (> 3000ms average)
- SQS DLQ messages (immediate alert)

### Dashboard Metrics
- Real-time API request volume
- Bedrock success rate and latency
- Hourly/daily cost tracking
- Cache hit ratio

## Architecture Decisions

### Why AWS Bedrock?
- Fully managed service (no infrastructure)
- Claude 3.5 Sonnet optimized for text analysis
- Built-in security and compliance
- Pay-per-use pricing model

### Why PHP?
- Requirement for backend PHP experience
- Mature AWS SDK support
- Strong ecosystem for APIs
- Suitable for high-volume processing

### Why SQS for Async Processing?
- Decouples processing from API requests
- Built-in retry mechanism
- Dead letter queue for failed jobs
- No infrastructure management

### Why Redis Caching?
- Sub-millisecond latency
- Reduces Bedrock API costs
- Simple key-value model
- Widely adopted standard

## Limitations & Future Enhancements

### Current Limitations
- Demo authentication (production needs proper user management)
- Single region deployment
- No multi-document batch API

### Planned Enhancements
- Database integration for metadata storage
- Real-time WebSocket updates for long-running analyses
- Multi-model support (compare Claude vs other LLMs)
- Advanced search capabilities with OpenSearch
- Document versioning and change tracking

## License

MIT License

## Author

Eduardo Duarte - Full Stack Engineer with focus on AI/ML integration and cloud architecture.
