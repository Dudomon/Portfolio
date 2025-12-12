# Architecture Decision Record (ADR)

## Context

This document records significant architectural decisions made for the Legal Document Analyzer platform, including rationale, alternatives considered, and trade-offs.

## ADR-001: AWS Bedrock as Primary LLM Provider

### Status
Accepted

### Context
Need to integrate LLM capabilities for legal document analysis with production-ready reliability, security, and cost predictability.

### Decision
Use AWS Bedrock with Claude 3.5 Sonnet as the primary model for all document analysis tasks.

### Rationale
- **Fully Managed Service**: No infrastructure management, automatic scaling, built-in security
- **Enterprise Grade**: SLA-backed service with 99.9% uptime
- **Compliance**: AWS compliance certifications suitable for public sector
- **Cost Model**: Pay-per-use with no upfront commitments
- **Model Quality**: Claude 3.5 Sonnet excels at structured output and complex reasoning
- **Integration**: Native AWS SDK support, seamless IAM integration

### Alternatives Considered

1. **OpenAI API**
   - Pros: Cutting-edge models, good documentation
   - Cons: Higher cost, external dependency, less AWS integration

2. **Self-hosted LLMs (vLLM, TGI)**
   - Pros: Full control, potentially lower cost at scale
   - Cons: Complex infrastructure, GPU management, scaling challenges

3. **Azure OpenAI Service**
   - Pros: Enterprise features, compliance
   - Cons: Multi-cloud complexity, less integration with existing AWS infrastructure

### Consequences
- **Positive**: Fast time to market, predictable costs, enterprise reliability
- **Negative**: Vendor lock-in to AWS, limited model customization
- **Mitigation**: Abstract Bedrock calls behind service interface for future provider flexibility

### Cost Analysis
- Input: $3.00 per million tokens
- Output: $15.00 per million tokens
- Average document (2K input + 500 output tokens): ~$0.0135 per analysis
- 10,000 analyses/month: ~$135/month (acceptable for MVP)

## ADR-002: PHP as Backend Language

### Status
Accepted

### Context
Need to demonstrate backend API development skills in PHP for job requirements while maintaining production quality.

### Decision
Implement backend in PHP 8.2 with modern practices (strict types, PSR-12, dependency injection).

### Rationale
- **Job Requirement**: Position explicitly requires PHP experience
- **Modern PHP**: PHP 8.2 offers strong typing, performance, and mature ecosystem
- **AWS SDK**: Official AWS SDK for PHP with complete Bedrock support
- **Ecosystem**: Rich ecosystem for API development (Composer, testing tools)

### Alternatives Considered

1. **Python with FastAPI**
   - Pros: Better ML ecosystem, async native
   - Cons: Doesn't demonstrate PHP skills required for position

2. **Node.js**
   - Pros: Excellent async support, JavaScript ecosystem
   - Cons: Not the required technology stack

### Consequences
- **Positive**: Directly demonstrates required skills, mature tooling
- **Negative**: PHP less common in ML/AI ecosystem
- **Trade-offs**: Some AWS features easier in Python, but PHP SDK is complete

## ADR-003: Asynchronous Processing with SQS

### Status
Accepted

### Context
Large documents (>100KB) can take 10-30 seconds to process with Bedrock. Synchronous API would timeout or provide poor UX.

### Decision
Use Amazon SQS for asynchronous processing of large documents and batch operations.

### Rationale
- **Decoupling**: API remains responsive, processing happens in background
- **Reliability**: Built-in retry mechanism, dead letter queue for failures
- **Scalability**: Queue depth automatically scales with load
- **Cost**: First 1M requests/month free, then $0.40 per million
- **Simple**: No infrastructure management, native AWS integration

### Implementation
```php
// For documents > 100KB
if (strlen($text) > 100000) {
    return $this->processAsync($request, 'analyze');
}
```

### Alternatives Considered

1. **Lambda + EventBridge**
   - Pros: Serverless, event-driven
   - Cons: More complex, cold starts, 15min Lambda limit

2. **Database polling (cronjob)**
   - Pros: Simple, no additional service
   - Cons: Higher latency, less scalable, polling overhead

### Consequences
- **Positive**: API stays fast (<200ms), handles any document size
- **Negative**: Complexity of async status checking
- **Monitoring**: Added CloudWatch metrics for queue depth and processing time

## ADR-004: Redis for Response Caching

### Status
Accepted

### Context
Identical or similar queries should not redundantly call Bedrock API, wasting cost and time.

### Decision
Implement Redis caching with SHA-256 hash-based cache keys for Bedrock responses.

### Rationale
- **Cost Savings**: Cache identical requests, reduce Bedrock costs by 30-50%
- **Performance**: Sub-millisecond cache hits vs seconds for Bedrock calls
- **TTL Control**: Configurable expiration (default 1 hour)
- **Simple**: Key-value model perfect for request/response caching

### Cache Key Strategy
```php
$cacheKey = 'bedrock:' . hash('sha256', json_encode([
    'prompt' => $prompt,
    'model_id' => $this->modelId,
    'options' => $options,
]));
```

### Alternatives Considered

1. **Application-level in-memory cache**
   - Pros: No external dependency
   - Cons: Lost on restart, not shared across instances

2. **ElastiCache Memcached**
   - Pros: Slightly faster
   - Cons: Less features than Redis, no persistence option

### Consequences
- **Positive**: Significant cost reduction, better performance
- **Negative**: Cache invalidation complexity, Redis dependency
- **Metrics**: Track cache hit/miss rates in CloudWatch

## ADR-005: JWT for Authentication

### Status
Accepted

### Context
Need secure, stateless authentication for RESTful API with role-based access control.

### Decision
Use JWT (JSON Web Tokens) with HS256 signing algorithm for authentication and authorization.

### Rationale
- **Stateless**: No server-side session storage needed
- **Scalable**: Works with multiple API instances
- **Standard**: Industry-standard with mature libraries
- **Payload**: Can include user roles and permissions
- **Expiration**: Built-in token expiration (configurable)

### Implementation
```php
$token = JWT::encode([
    'user_id' => $user['id'],
    'roles' => $user['roles'],
    'exp' => time() + 3600,
], $secret, 'HS256');
```

### Alternatives Considered

1. **Session-based authentication**
   - Pros: Server controls revocation
   - Cons: Not stateless, scaling complexity

2. **OAuth 2.0**
   - Pros: Industry standard for third-party auth
   - Cons: Overkill for simple API, complex implementation

### Consequences
- **Positive**: Simple, scalable, standard
- **Negative**: Token revocation requires additional mechanism
- **Security**: Short expiration (1 hour), refresh token support

## ADR-006: Comprehensive Observability Strategy

### Status
Accepted

### Context
Production system needs visibility into performance, costs, errors, and usage patterns.

### Decision
Implement multi-layered observability with CloudWatch Logs, Metrics, X-Ray tracing, and custom dashboards.

### Rationale
- **Debugging**: Structured logs with request correlation
- **Performance**: Track latency at every layer
- **Cost Monitoring**: Track Bedrock usage and costs in real-time
- **Alerting**: Proactive alerts for errors and anomalies
- **Compliance**: Audit trail for all operations

### Observability Layers

1. **Logging (CloudWatch Logs)**
```php
$this->observability->log('info', 'Bedrock API invocation', [
    'model_id' => $this->modelId,
    'latency_ms' => $latency,
    'input_tokens' => $inputTokens,
    'output_tokens' => $outputTokens,
    'cost_usd' => $cost,
]);
```

2. **Metrics (CloudWatch Metrics)**
- Request counts (success/error)
- Latency (P50, P90, P99)
- Cost per request
- Token usage
- Cache hit/miss rates

3. **Tracing (X-Ray)**
- End-to-end request flow
- Service dependency map
- Performance bottleneck identification

4. **Dashboard**
- Real-time API metrics
- Bedrock performance and costs
- Queue depth and processing time

### Alternatives Considered

1. **ELK Stack (Elasticsearch, Logstash, Kibana)**
   - Pros: Powerful search, flexible
   - Cons: Infrastructure overhead, cost

2. **Datadog / New Relic**
   - Pros: Better UI, advanced features
   - Cons: Additional cost, third-party dependency

### Consequences
- **Positive**: Complete visibility, proactive issue detection
- **Negative**: CloudWatch costs (mitigated with retention policies)
- **ROI**: Faster debugging saves more than monitoring costs

## ADR-007: KMS Encryption for Documents

### Status
Accepted

### Context
Legal documents contain sensitive information requiring encryption at rest.

### Decision
Use AWS KMS (Key Management Service) for S3 bucket encryption with customer-managed keys.

### Rationale
- **Compliance**: Required for public sector data
- **Key Rotation**: Automatic key rotation support
- **Access Control**: Fine-grained IAM policies
- **Audit Trail**: CloudTrail logs all key usage
- **Integration**: Native S3 integration

### Implementation
```terraform
resource "aws_s3_bucket_server_side_encryption_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.documents.arn
    }
  }
}
```

### Alternatives Considered

1. **S3 default encryption (SSE-S3)**
   - Pros: Simpler, no KMS cost
   - Cons: Less control, no rotation, weaker compliance

2. **Client-side encryption**
   - Pros: Maximum security
   - Cons: Complex key management, performance overhead

### Consequences
- **Positive**: Compliance-ready, audit trail, key rotation
- **Negative**: KMS costs ($1/key/month + $0.03 per 10K requests)
- **Trade-off**: Small cost for significant security improvement

## ADR-008: Terraform for Infrastructure

### Status
Accepted

### Context
Infrastructure needs to be reproducible, version-controlled, and reviewable.

### Decision
Use Terraform for infrastructure as code with S3 backend and state locking.

### Rationale
- **Version Control**: Infrastructure changes tracked in Git
- **Reproducible**: Identical infra across environments
- **Review**: Pull requests for infrastructure changes
- **State Management**: S3 backend with DynamoDB locking
- **Multi-Provider**: Not locked into CloudFormation

### Infrastructure Components
```
terraform/
├── main.tf       # Resources (S3, SQS, KMS, IAM)
├── variables.tf  # Configurable inputs
├── outputs.tf    # Important values
└── backend.tf    # State management
```

### Alternatives Considered

1. **AWS CloudFormation**
   - Pros: Native AWS, no state file
   - Cons: AWS-only, verbose YAML

2. **AWS CDK**
   - Pros: Code-based, strong typing
   - Cons: CloudFormation under the hood, learning curve

3. **Manual Console**
   - Pros: Quick for prototyping
   - Cons: Not reproducible, error-prone

### Consequences
- **Positive**: Professional DevOps practices, reviewable changes
- **Negative**: Learning curve, state file management
- **Best Practice**: Aligns with enterprise infrastructure management

## ADR-009: Rate Limiting Strategy

### Status
Accepted

### Context
API needs protection from abuse while ensuring fair usage for legitimate clients.

### Decision
Implement token bucket rate limiting with Redis, 100 requests per minute per client.

### Rationale
- **Protection**: Prevent abuse and DoS attacks
- **Fair Usage**: Ensure resources available for all users
- **Cost Control**: Limit maximum Bedrock API costs
- **Feedback**: Return rate limit headers to clients

### Implementation
```php
public function handle(Request $request, callable $next): Response
{
    $identifier = $this->getIdentifier($request);
    $key = "rate_limit:{$identifier}";

    $current = (int) $this->cache->get($key) ?: 0;

    if ($current >= $this->maxRequests) {
        return Response::error('Rate limit exceeded', 429);
    }

    $this->cache->set($key, $current + 1, $this->windowSeconds);
    // ...
}
```

### Rate Limit Headers
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 73
X-RateLimit-Reset: 1234567890
```

### Alternatives Considered

1. **API Gateway throttling**
   - Pros: Managed service
   - Cons: Less flexible, additional cost

2. **Fixed window**
   - Pros: Simpler implementation
   - Cons: Burst issues at window boundaries

### Consequences
- **Positive**: API protection, predictable costs
- **Negative**: Legitimate high-volume users may hit limits
- **Monitoring**: Track rate limit violations in CloudWatch

## ADR-010: Structured Error Handling

### Status
Accepted

### Context
Errors need to be handled consistently with appropriate logging and user feedback.

### Decision
Implement middleware-based error handling with structured error responses and comprehensive logging.

### Rationale
- **Consistency**: All errors follow same format
- **Security**: No sensitive data leaked in responses
- **Debugging**: Full error details in logs
- **Client Experience**: Clear, actionable error messages

### Error Response Format
```json
{
  "success": false,
  "message": "Error description",
  "errors": {},
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Debug Mode
```php
if ($_ENV['APP_DEBUG'] === 'true') {
    $errorData['file'] = $e->getFile();
    $errorData['line'] = $e->getLine();
    $errorData['trace'] = explode("\n", $e->getTraceAsString());
}
```

### Alternatives Considered

1. **Default PHP error handling**
   - Pros: Simple
   - Cons: Inconsistent, exposes internals

2. **Exception-only (no middleware)**
   - Pros: Simpler flow
   - Cons: Scattered error handling, inconsistent

### Consequences
- **Positive**: Better debugging, secure, consistent
- **Negative**: Additional middleware layer
- **Security**: Debug mode disabled in production

## Performance Targets

### API Response Times
- Authentication: < 100ms
- Document upload: < 500ms
- Small document analysis (sync): < 3s
- Large document analysis (async): < 30s background

### Bedrock Integration
- P50 latency: ~1.5s
- P90 latency: ~3s
- P99 latency: ~5s

### Cache Performance
- Redis hit: < 5ms
- Cache hit rate target: > 40%

### Cost Targets
- Per document analysis: ~$0.01-0.02
- Monthly (10K analyses): ~$100-200
- Cache reduces costs by 30-50%

## Security Considerations

### Data Protection
- Documents encrypted at rest (KMS)
- TLS 1.2+ in transit
- Presigned URLs for time-limited access
- Public S3 access blocked

### Authentication
- JWT with 1-hour expiration
- Refresh token support
- Role-based access control
- Secure password hashing (bcrypt)

### API Security
- Input validation and sanitization
- Rate limiting (100 req/min)
- Request size limits (10MB max)
- No sensitive data in error responses

### AWS Security
- IAM least privilege policies
- KMS encryption for sensitive data
- CloudTrail for audit logs
- VPC isolation (production)

### Compliance
- GDPR: Data encryption, right to deletion
- LGPD (Brazilian): Data protection controls
- Public Sector: Audit trail, access controls

## Scalability Considerations

### Horizontal Scaling
- Stateless API design
- JWT removes session state
- Redis for shared cache
- S3 for unlimited storage

### Vertical Scaling
- PHP-FPM process management
- Redis memory scaling
- SQS queue capacity

### Cost Scaling
- Bedrock: Pay per token (no minimum)
- S3: Pay per GB stored
- SQS: First 1M requests free
- CloudWatch: Free tier generous

### Bottlenecks
- Bedrock API rate limits (handled by SQS queue)
- Redis memory (monitoring + scaling plan)
- S3 request rates (non-issue at expected scale)

## Future Architecture Considerations

### Multi-Region
- S3 cross-region replication
- DynamoDB for distributed sessions
- Route 53 for traffic management

### Database Integration
- RDS PostgreSQL for metadata
- Document relationships
- Search indexing

### Advanced Features
- WebSocket for real-time updates
- OpenSearch for document search
- Step Functions for complex workflows

### Multi-Model Support
- Abstract LLM interface
- Support multiple providers
- Cost/quality optimization

## Conclusion

This architecture balances production-readiness, cost-efficiency, and demonstration of required skills (PHP, AWS Bedrock, high-volume APIs, observability). Each decision considers trade-offs and includes mitigation strategies for limitations.
