<?php

declare(strict_types=1);

namespace MultiLLMGateway\Providers;

use Aws\BedrockRuntime\BedrockRuntimeClient;
use Aws\Exception\AwsException;
use MultiLLMGateway\Services\ObservabilityService;

class BedrockProvider implements LLMProviderInterface
{
    private BedrockRuntimeClient $client;
    private ObservabilityService $observability;
    private string $modelId;
    private array $stats = [];

    public function __construct(ObservabilityService $observability)
    {
        $this->client = new BedrockRuntimeClient([
            'region' => $_ENV['AWS_REGION'],
            'version' => 'latest',
            'credentials' => [
                'key' => $_ENV['AWS_ACCESS_KEY_ID'],
                'secret' => $_ENV['AWS_SECRET_ACCESS_KEY'],
            ],
        ]);

        $this->observability = $observability;
        $this->modelId = $_ENV['BEDROCK_MODEL_ID'];
        $this->loadStats();
    }

    public function getName(): string
    {
        return 'bedrock';
    }

    public function invoke(string $prompt, array $options = []): array
    {
        $startTime = microtime(true);

        try {
            $payload = [
                'anthropic_version' => 'bedrock-2023-05-31',
                'max_tokens' => $options['max_tokens'] ?? 4096,
                'temperature' => $options['temperature'] ?? 0.7,
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

            $result = $this->client->invokeModel([
                'modelId' => $this->modelId,
                'contentType' => 'application/json',
                'accept' => 'application/json',
                'body' => json_encode($payload),
            ]);

            $response = json_decode($result->get('body')->getContents(), true);

            $latency = (microtime(true) - $startTime) * 1000;
            $inputTokens = $response['usage']['input_tokens'] ?? 0;
            $outputTokens = $response['usage']['output_tokens'] ?? 0;
            $cost = $this->getEstimatedCost($inputTokens, $outputTokens);

            $this->recordSuccess($latency, $cost);
            $this->observability->endTrace('bedrock.invoke');

            return [
                'provider' => 'bedrock',
                'model' => $this->modelId,
                'content' => $response['content'][0]['text'] ?? '',
                'usage' => [
                    'input_tokens' => $inputTokens,
                    'output_tokens' => $outputTokens,
                ],
                'cost' => $cost,
                'latency' => $latency,
                'success' => true,
            ];

        } catch (AwsException $e) {
            $this->recordError();
            $this->observability->log('error', 'Bedrock provider error', [
                'error' => $e->getMessage(),
                'code' => $e->getAwsErrorCode(),
            ]);
            $this->observability->endTrace('bedrock.invoke');

            throw $e;
        }
    }

    public function streamInvoke(string $prompt, array $options = []): \Generator
    {
        $payload = [
            'anthropic_version' => 'bedrock-2023-05-31',
            'max_tokens' => $options['max_tokens'] ?? 4096,
            'temperature' => $options['temperature'] ?? 0.7,
            'messages' => [
                [
                    'role' => 'user',
                    'content' => $prompt,
                ],
            ],
        ];

        try {
            $result = $this->client->invokeModelWithResponseStream([
                'modelId' => $this->modelId,
                'contentType' => 'application/json',
                'accept' => 'application/json',
                'body' => json_encode($payload),
            ]);

            foreach ($result['body'] as $event) {
                if (isset($event['chunk'])) {
                    $chunk = json_decode($event['chunk']->getContents(), true);

                    if (isset($chunk['type']) && $chunk['type'] === 'content_block_delta') {
                        yield $chunk['delta']['text'] ?? '';
                    }
                }
            }

        } catch (AwsException $e) {
            $this->recordError();
            throw $e;
        }
    }

    public function isAvailable(): bool
    {
        return $this->stats['error_rate'] < 0.5;
    }

    public function getEstimatedCost(int $inputTokens, int $outputTokens): float
    {
        $inputCostPer1k = 0.003;
        $outputCostPer1k = 0.015;

        $inputCost = ($inputTokens / 1000) * $inputCostPer1k;
        $outputCost = ($outputTokens / 1000) * $outputCostPer1k;

        return round($inputCost + $outputCost, 6);
    }

    public function getAverageLatency(): float
    {
        return $this->stats['avg_latency'] ?? 0.0;
    }

    public function getErrorRate(): float
    {
        return $this->stats['error_rate'] ?? 0.0;
    }

    public function healthCheck(): bool
    {
        try {
            $this->invoke('test', ['max_tokens' => 10]);
            return true;
        } catch (\Exception $e) {
            return false;
        }
    }

    private function recordSuccess(float $latency, float $cost): void
    {
        $this->stats['total_requests'] = ($this->stats['total_requests'] ?? 0) + 1;
        $this->stats['successful_requests'] = ($this->stats['successful_requests'] ?? 0) + 1;
        $this->stats['total_latency'] = ($this->stats['total_latency'] ?? 0) + $latency;
        $this->stats['total_cost'] = ($this->stats['total_cost'] ?? 0) + $cost;
        $this->stats['avg_latency'] = $this->stats['total_latency'] / $this->stats['successful_requests'];
        $this->stats['error_rate'] = 1 - ($this->stats['successful_requests'] / $this->stats['total_requests']);

        $this->saveStats();

        $this->observability->recordMetric('bedrock.request.success', 1);
        $this->observability->recordMetric('bedrock.latency', $latency);
        $this->observability->recordMetric('bedrock.cost', $cost);
    }

    private function recordError(): void
    {
        $this->stats['total_requests'] = ($this->stats['total_requests'] ?? 0) + 1;
        $this->stats['error_rate'] = 1 - (($this->stats['successful_requests'] ?? 0) / $this->stats['total_requests']);

        $this->saveStats();

        $this->observability->recordMetric('bedrock.request.error', 1);
    }

    private function loadStats(): void
    {
        $this->stats = [
            'total_requests' => 0,
            'successful_requests' => 0,
            'total_latency' => 0,
            'total_cost' => 0,
            'avg_latency' => 0,
            'error_rate' => 0,
        ];
    }

    private function saveStats(): void
    {
    }
}
