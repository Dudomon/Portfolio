<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Services;

use Aws\BedrockRuntime\BedrockRuntimeClient;
use Aws\Exception\AwsException;
use LegalDocAnalyzer\Services\CacheService;
use LegalDocAnalyzer\Services\ObservabilityService;
use LegalDocAnalyzer\Exceptions\BedrockException;

class BedrockService
{
    private BedrockRuntimeClient $client;
    private CacheService $cache;
    private ObservabilityService $observability;
    private string $modelId;
    private int $maxTokens;
    private float $temperature;

    public function __construct(
        CacheService $cache,
        ObservabilityService $observability
    ) {
        $this->client = new BedrockRuntimeClient([
            'region' => $_ENV['AWS_REGION'],
            'version' => 'latest',
            'credentials' => [
                'key' => $_ENV['AWS_ACCESS_KEY_ID'],
                'secret' => $_ENV['AWS_SECRET_ACCESS_KEY'],
            ],
        ]);

        $this->cache = $cache;
        $this->observability = $observability;
        $this->modelId = $_ENV['BEDROCK_MODEL_ID'];
        $this->maxTokens = (int) $_ENV['BEDROCK_MAX_TOKENS'];
        $this->temperature = (float) $_ENV['BEDROCK_TEMPERATURE'];
    }

    public function invoke(string $prompt, array $options = []): array
    {
        $startTime = microtime(true);
        $cacheKey = $this->getCacheKey($prompt, $options);

        if ($cachedResponse = $this->cache->get($cacheKey)) {
            $this->observability->recordMetric('bedrock.cache.hit', 1);
            return $cachedResponse;
        }

        $this->observability->recordMetric('bedrock.cache.miss', 1);

        try {
            $payload = [
                'anthropic_version' => 'bedrock-2023-05-31',
                'max_tokens' => $options['max_tokens'] ?? $this->maxTokens,
                'temperature' => $options['temperature'] ?? $this->temperature,
                'messages' => [
                    [
                        'role' => 'user',
                        'content' => $prompt,
                    ],
                ],
            ];

            if (isset($options['system'])) {
                $payload['system'] = $options['system'];
            }

            $this->observability->startTrace('bedrock.invoke');
            $this->observability->log('info', 'Invoking Bedrock API', [
                'model_id' => $this->modelId,
                'prompt_length' => strlen($prompt),
            ]);

            $result = $this->client->invokeModel([
                'modelId' => $this->modelId,
                'contentType' => 'application/json',
                'accept' => 'application/json',
                'body' => json_encode($payload),
            ]);

            $response = json_decode($result->get('body')->getContents(), true);

            $latency = (microtime(true) - $startTime) * 1000;
            $this->observability->recordMetric('bedrock.latency', $latency);
            $this->observability->recordMetric('bedrock.request.success', 1);
            $this->observability->endTrace('bedrock.invoke');

            $inputTokens = $response['usage']['input_tokens'] ?? 0;
            $outputTokens = $response['usage']['output_tokens'] ?? 0;

            $this->observability->recordMetric('bedrock.tokens.input', $inputTokens);
            $this->observability->recordMetric('bedrock.tokens.output', $outputTokens);

            $cost = $this->calculateCost($inputTokens, $outputTokens);
            $this->observability->recordMetric('bedrock.cost', $cost);

            $this->observability->log('info', 'Bedrock API invocation completed', [
                'model_id' => $this->modelId,
                'latency_ms' => $latency,
                'input_tokens' => $inputTokens,
                'output_tokens' => $outputTokens,
                'cost_usd' => $cost,
            ]);

            $result = [
                'content' => $response['content'][0]['text'] ?? '',
                'usage' => [
                    'input_tokens' => $inputTokens,
                    'output_tokens' => $outputTokens,
                ],
                'cost' => $cost,
                'latency' => $latency,
            ];

            $this->cache->set($cacheKey, $result);

            return $result;

        } catch (AwsException $e) {
            $this->observability->recordMetric('bedrock.request.error', 1);
            $this->observability->log('error', 'Bedrock API error', [
                'error_code' => $e->getAwsErrorCode(),
                'error_message' => $e->getMessage(),
                'model_id' => $this->modelId,
            ]);
            $this->observability->endTrace('bedrock.invoke');

            throw new BedrockException(
                "Bedrock API error: {$e->getMessage()}",
                $e->getCode(),
                $e
            );
        }
    }

    private function getCacheKey(string $prompt, array $options): string
    {
        $data = [
            'prompt' => $prompt,
            'model_id' => $this->modelId,
            'options' => $options,
        ];

        return 'bedrock:' . hash('sha256', json_encode($data));
    }

    private function calculateCost(int $inputTokens, int $outputTokens): float
    {
        $inputCostPer1k = 0.003;
        $outputCostPer1k = 0.015;

        $inputCost = ($inputTokens / 1000) * $inputCostPer1k;
        $outputCost = ($outputTokens / 1000) * $outputCostPer1k;

        return round($inputCost + $outputCost, 6);
    }

    public function extractEntities(string $text, string $documentType): array
    {
        $prompt = $this->buildEntityExtractionPrompt($text, $documentType);

        $result = $this->invoke($prompt, [
            'system' => 'You are an expert legal document analyzer specializing in entity extraction for Brazilian public sector documents.',
        ]);

        return json_decode($result['content'], true) ?? [];
    }

    public function classify(string $text): array
    {
        $prompt = $this->buildClassificationPrompt($text);

        $result = $this->invoke($prompt, [
            'system' => 'You are an expert legal document classifier for Brazilian public sector.',
        ]);

        return json_decode($result['content'], true) ?? [];
    }

    public function summarize(string $text, int $maxLength = 500): array
    {
        $prompt = $this->buildSummarizationPrompt($text, $maxLength);

        $result = $this->invoke($prompt, [
            'system' => 'You are an expert legal document summarizer.',
        ]);

        return [
            'summary' => $result['content'],
            'usage' => $result['usage'],
            'cost' => $result['cost'],
        ];
    }

    private function buildEntityExtractionPrompt(string $text, string $documentType): string
    {
        return <<<PROMPT
Extract all relevant entities from the following {$documentType} document.

Return a JSON object with these fields:
- people: array of person names mentioned
- organizations: array of organization names
- locations: array of locations
- dates: array of important dates
- laws: array of referenced laws or regulations
- case_numbers: array of case or process numbers
- monetary_values: array of monetary values mentioned

Document text:
{$text}

Return ONLY valid JSON, no additional text.
PROMPT;
    }

    private function buildClassificationPrompt(string $text): string
    {
        return <<<PROMPT
Classify the following legal document into one of these categories:
- sentenca (court decision)
- peticao (petition)
- contrato (contract)
- lei (law/regulation)
- edital (public notice)
- ata (minutes)
- parecer (legal opinion)
- despacho (dispatch/order)
- outro (other)

Also determine:
- urgency: baixa, media, alta
- complexity: baixa, media, alta
- requires_review: true/false

Return a JSON object with: category, subcategory (if applicable), urgency, complexity, requires_review, confidence_score

Document text:
{$text}

Return ONLY valid JSON, no additional text.
PROMPT;
    }

    private function buildSummarizationPrompt(string $text, int $maxLength): string
    {
        return <<<PROMPT
Summarize the following legal document in Portuguese. The summary should be no more than {$maxLength} words.

Focus on:
- Main subject or purpose
- Key parties involved
- Important decisions or conclusions
- Critical dates or deadlines
- Required actions

Document text:
{$text}

Provide a clear, professional summary in Portuguese.
PROMPT;
    }
}
