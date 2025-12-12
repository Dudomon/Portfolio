# Multi-LLM Gateway with LLMOps & Model Context Protocol

Production-grade intelligent gateway for multiple LLM providers (AWS Bedrock, OpenAI, Google Gemini) with cost optimization, automatic fallback, quality monitoring, and Model Context Protocol (MCP) support. Built with PHP 8.2.

## Key Features

- **Multi-Provider Support**: Bedrock (Claude), OpenAI (GPT-4), Google (Gemini)
- **Intelligent Routing**: Cost-optimized, latency-aware, reliability-based
- **Automatic Fallback**: Seamless failover between providers
- **Model Context Protocol (MCP)**: Standardized tool calling and context management
- **LLMOps**: Quality evaluation with LLM-as-judge, anomaly detection
- **Cost Tracking**: Real-time cost monitoring across all providers
- **Quality Monitoring**: Automated response quality scoring
- **Comprehensive Observability**: Metrics, logging, tracing

## Architecture

```
┌────────────────────────────────────────────┐
│           Gateway Service                   │
│  ┌──────────────────────────────────────┐  │
│  │    Intelligent Router                │  │
│  │  (Cost/Latency/Reliability)          │  │
│  └──────────────┬───────────────────────┘  │
│                 │                           │
│  ┌──────────────┴───────────────────────┐  │
│  │       Provider Abstraction            │  │
│  └──┬────────┬────────┬─────────────────┘  │
│     │        │        │                     │
└─────┼────────┼────────┼─────────────────────┘
      │        │        │
      ▼        ▼        ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Bedrock  │ │ OpenAI   │ │ Gemini   │
│ Claude   │ │ GPT-4    │ │ Pro      │
└──────────┘ └──────────┘ └──────────┘
      │        │        │
      └────────┴────────┘
             │
      ┌──────▼──────┐
      │   LLMOps    │
      │  - Quality  │
      │  - Cost     │
      │  - Metrics  │
      └─────────────┘
```

## Core Components

### 1. Provider Abstraction (Driver Pattern)

Unified interface for all LLM providers:

```php
interface LLMProviderInterface
{
    public function invoke(string $prompt, array $options = []): array;
    public function streamInvoke(string $prompt, array $options = []): \Generator;
    public function isAvailable(): bool;
    public function getEstimatedCost(int $inputTokens, int $outputTokens): float;
    public function getAverageLatency(): float;
    public function getErrorRate(): float;
}
```

**Implementations:**
- BedrockProvider: Claude 3.5 Sonnet
- OpenAIProvider: GPT-4 Turbo
- GeminiProvider: Gemini 1.5 Pro (easily extensible)

### 2. Intelligent Routing

**Cost-Optimized Router** (default):
- Estimates token count from prompt
- Calculates cost for each provider
- Weighs cost (60%), latency (20%), reliability (20%)
- Selects optimal provider dynamically

```php
$provider = $router->selectProvider($providers, $prompt);
// Automatically selects cheapest reliable provider
```

**Alternative Strategies:**
- Latency-optimized: Prioritizes fastest provider
- Round-robin: Load balancing across providers
- Manual: Specific provider selection

### 3. Automatic Fallback

Seamless failover when primary provider fails:

```php
try {
    $response = $primaryProvider->invoke($prompt);
} catch (Exception $e) {
    // Automatic fallback to next available provider
    $response = $fallbackProvider->invoke($prompt);
    $response['fallback'] = true;
}
```

**Fallback Logic:**
- Attempts all available providers in order
- Skips unavailable providers (high error rate)
- Tracks all attempted providers
- Logs failures for analysis

### 4. Model Context Protocol (MCP)

Standardized protocol for tool calling and context management:

**MCP Context:**
```php
$context = new MCPContext('2024-11-05');

$context->addTool('analyze_sentiment', 'Analyze sentiment of text', [
    'text' => ['type' => 'string', 'required' => true],
]);

$context->addPrompt('legal_analysis', 'Structured legal document analysis', [
    ['name' => 'document_type', 'required' => true],
]);
```

**MCP Handler:**
```php
$mcpHandler = new MCPHandler($observability);

// Augment prompts with available tools
$augmentedPrompt = $mcpHandler->augmentPrompt($prompt, $mcpContext);

// Process tool calls in responses
$processedResponse = $mcpHandler->processResponse($response);
```

**Benefits:**
- Standardized tool interface across providers
- Automatic tool discovery
- Context injection for better responses
- Tool execution tracking

### 5. LLMOps - Quality Evaluation

**LLM-as-Judge Pattern:**
```php
$evaluator = new QualityEvaluator($observability, $judgeProvider);

$quality = $evaluator->evaluateResponse($prompt, $response, [
    'relevance' => 'Is response relevant to prompt?',
    'accuracy' => 'Is information accurate?',
    'completeness' => 'Does it fully address the prompt?',
    'clarity' => 'Is it clear and well-structured?',
]);

// Returns:
// {
//   "overall_score": 8.5,
//   "criteria_scores": {
//     "relevance": 9,
//     "accuracy": 8,
//     "completeness": 8,
//     "clarity": 9
//   }
// }
```

**Anomaly Detection:**
```php
$anomalies = $evaluator->detectAnomalies($responses);

// Detects:
// - High latency (>10s)
// - High cost (>$0.10)
// - Short responses (<10 chars)
// - Failures
```

**Provider Comparison:**
```php
$results = $evaluator->compareProviders($prompt, $providers);

// Compare quality across all providers for given prompt
// Useful for A/B testing and provider selection
```

### 6. Gateway Service

Central orchestration:

```php
$gateway = new GatewayService($router, $mcpHandler, $observability, $qualityEvaluator);

// Register providers
$gateway->registerProvider(new BedrockProvider($observability));
$gateway->registerProvider(new OpenAIProvider($observability));

// Invoke with automatic routing, fallback, and quality monitoring
$response = $gateway->invoke($prompt, [
    'use_mcp' => true,
    'mcp_context' => ['tools' => ['analyze_sentiment']],
]);

// Response includes:
// - content: LLM response
// - provider: Which provider was used
// - cost: Actual cost in USD
// - latency: Response time in ms
// - quality: Quality scores (if enabled)
// - fallback: Whether fallback was used
```

## Installation

```bash
composer install
cp .env.example .env
# Configure AWS, OpenAI, Google credentials
```

## Configuration

### Provider Configuration

```env
# AWS Bedrock
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0

# OpenAI
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4-turbo-preview

# Google Gemini
GOOGLE_API_KEY=your-key
GOOGLE_MODEL=gemini-1.5-pro
```

### Gateway Settings

```env
ROUTING_STRATEGY=cost_optimized    # cost_optimized, latency_optimized, round_robin
ENABLE_FALLBACK=true                # Automatic failover
ENABLE_QUALITY_MONITORING=true      # LLM-as-judge evaluation

COST_THRESHOLD_USD=0.10             # Alert if cost exceeds threshold
LATENCY_THRESHOLD_MS=5000           # Alert if latency exceeds threshold
```

## Usage Examples

### Basic Invocation

```php
use MultiLLMGateway\Services\GatewayService;

$gateway = new GatewayService($router, $mcpHandler, $observability);
$gateway->registerProvider(new BedrockProvider($observability));
$gateway->registerProvider(new OpenAIProvider($observability));

$response = $gateway->invoke("Explain quantum computing in simple terms", [
    'max_tokens' => 500,
    'temperature' => 0.7,
]);

echo $response['content'];
echo "Cost: ${$response['cost']} USD\n";
echo "Latency: {$response['latency']}ms\n";
echo "Provider: {$response['provider']}\n";
```

### Streaming Response

```php
foreach ($gateway->streamInvoke($prompt) as $chunk) {
    echo $chunk;
    flush();
}
```

### With MCP Context

```php
$response = $gateway->invoke($prompt, [
    'use_mcp' => true,
    'mcp_context' => [
        'tools' => ['analyze_sentiment', 'extract_entities'],
        'resources' => [
            ['uri' => 'file://document.pdf', 'name' => 'Contract'],
        ],
    ],
]);
```

### Quality Evaluation

```php
$quality = $evaluator->evaluateResponse($prompt, $response['content']);

if ($quality['overall_score'] < 5) {
    // Low quality, may want to retry with different provider
    $response = $gateway->invoke($prompt, ['provider' => 'openai']);
}
```

### Provider Health Check

```php
$health = $gateway->healthCheck();

// Returns:
// {
//   "bedrock": {"healthy": true, "available": true, "error_rate": 0.02},
//   "openai": {"healthy": true, "available": true, "error_rate": 0.01}
// }
```

## Cost Analysis

### Provider Pricing (per million tokens)

| Provider | Input | Output | Example Cost (2K in, 500 out) |
|----------|-------|--------|-------------------------------|
| Bedrock (Claude) | $3.00 | $15.00 | $0.0135 |
| OpenAI (GPT-4) | $10.00 | $30.00 | $0.0350 |
| Gemini Pro | $0.35 | $1.05 | $0.0012 |

### Cost Optimization

Gateway automatically routes to cheapest reliable provider:
- Gemini: Cheapest (85% cost reduction vs GPT-4)
- Bedrock: Middle ground (60% cost reduction vs GPT-4)
- OpenAI: Highest quality but most expensive

**Estimated Savings:**
- 10K requests/month with intelligent routing: ~$200 saved vs always using GPT-4

## LLMOps Metrics

### Tracked Metrics

**Per Provider:**
- `{provider}.request.success` - Successful requests
- `{provider}.request.error` - Failed requests
- `{provider}.latency` - Response latency (ms)
- `{provider}.cost` - Cost per request (USD)

**Gateway Level:**
- `gateway.request.success` - Successful gateway calls
- `gateway.request.error` - Failed gateway calls
- `gateway.fallback.success` - Successful fallbacks
- `router.selection` - Provider selection count
- `router.estimated_cost` - Estimated cost before invocation

**Quality:**
- `quality.score` - Overall quality score
- `quality.anomalies` - Number of anomalies detected
- `mcp.tool.success` - Successful tool executions
- `mcp.tool.error` - Failed tool executions

### Observability

**Structured Logging:**
```json
{
  "level": "info",
  "message": "Gateway invocation successful",
  "provider": "bedrock",
  "latency": 1523.45,
  "cost": 0.0135,
  "quality_score": 8.5,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Distributed Tracing:**
- `gateway.invoke` - End-to-end request
- `{provider}.invoke` - Provider-specific call
- `quality.evaluation` - Quality scoring
- `mcp.tool.{name}` - Tool execution

## Model Context Protocol (MCP) Specification

### MCP Version
Implements **MCP 2024-11-05** specification.

### Capabilities

**Tools:**
```json
{
  "name": "analyze_sentiment",
  "description": "Analyze sentiment of given text",
  "inputSchema": {
    "type": "object",
    "properties": {
      "text": {"type": "string", "description": "Text to analyze"}
    },
    "required": ["text"]
  }
}
```

**Resources:**
```json
{
  "uri": "file://document.pdf",
  "name": "Legal Contract",
  "mimeType": "application/pdf",
  "description": "Contract for review"
}
```

**Prompts:**
```json
{
  "name": "legal_analysis",
  "description": "Structured legal document analysis",
  "arguments": [
    {"name": "document_type", "required": true}
  ]
}
```

### Tool Execution

1. Gateway augments prompt with available tools
2. LLM response includes tool calls: `<tool_use>{"name": "analyze_sentiment", "arguments": {...}}</tool_use>`
3. MCPHandler executes tool and injects result
4. Final response includes tool execution results

## Advanced Features

### Custom Router Strategy

```php
class LatencyOptimizedRouter implements RouterInterface
{
    public function selectProvider(array $providers, string $prompt, array $options): LLMProviderInterface
    {
        // Custom logic: select fastest provider
        usort($providers, fn($a, $b) => $a->getAverageLatency() <=> $b->getAverageLatency());
        return $providers[0];
    }
}
```

### Custom MCP Tools

```php
$mcpHandler->registerTool('custom_analysis', function($args) {
    // Custom tool implementation
    return [
        'result' => 'Analysis complete',
        'score' => 0.85,
    ];
});
```

### Provider Comparison

```php
$results = $qualityEvaluator->compareProviders($prompt, [
    new BedrockProvider($observability),
    new OpenAIProvider($observability),
]);

// Compare quality and cost across providers
// Useful for selecting best provider for specific use case
```

## Architecture Decisions

### Why Driver Pattern for Providers?
- **Flexibility**: Easy to add new LLM providers
- **Abstraction**: Business logic doesn't depend on specific provider
- **Testing**: Mock providers for unit tests
- **Migration**: Switch providers without code changes

### Why Intelligent Routing?
- **Cost Optimization**: Automatically use cheapest reliable option
- **Performance**: Consider latency for time-sensitive requests
- **Reliability**: Avoid providers with high error rates
- **Flexibility**: Different strategies for different use cases

### Why MCP?
- **Standardization**: Industry-standard protocol for tool calling
- **Interoperability**: Works across different LLM providers
- **Future-Proof**: Designed for evolving LLM capabilities
- **Tool Discovery**: Automatic tool availability in prompts

### Why LLM-as-Judge?
- **Automated Quality**: No manual review needed
- **Consistency**: Same criteria applied to all responses
- **Scalability**: Evaluate millions of responses
- **Continuous Monitoring**: Detect quality degradation early

## Limitations & Future Work

### Current Limitations
- Google Gemini provider not fully implemented (interface ready)
- Quality evaluation adds latency (~2s)
- No persistent storage for metrics (only CloudWatch)

### Planned Enhancements
- Prompt caching for cost reduction
- Semantic cache (similar prompts)
- Database integration for metrics history
- Web dashboard for monitoring
- Automatic prompt optimization
- Multi-turn conversation support
- Fine-tuning integration

## Testing

```bash
composer run-script test     # PHPUnit tests
composer run-script stan     # Static analysis
composer run-script cs       # Code style
composer run-script check    # All checks
```

## License

MIT License

## Author

Eduardo Duarte - Specializing in LLM integration, multi-provider architectures, and production-grade AI systems.
