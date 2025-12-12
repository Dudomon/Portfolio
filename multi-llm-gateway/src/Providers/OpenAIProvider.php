<?php

declare(strict_types=1);

namespace MultiLLMGateway\Providers;

use GuzzleHttp\Client;
use GuzzleHttp\Exception\GuzzleException;
use MultiLLMGateway\Services\ObservabilityService;

class OpenAIProvider implements LLMProviderInterface
{
    private Client $client;
    private ObservabilityService $observability;
    private string $model;
    private string $apiKey;
    private array $stats = [];

    public function __construct(ObservabilityService $observability)
    {
        $this->client = new Client([
            'base_uri' => $_ENV['OPENAI_ENDPOINT'],
            'timeout' => 60,
        ]);

        $this->observability = $observability;
        $this->model = $_ENV['OPENAI_MODEL'];
        $this->apiKey = $_ENV['OPENAI_API_KEY'];
        $this->loadStats();
    }

    public function getName(): string
    {
        return 'openai';
    }

    public function invoke(string $prompt, array $options = []): array
    {
        $startTime = microtime(true);

        try {
            $this->observability->startTrace('openai.invoke');

            $response = $this->client->post('/chat/completions', [
                'headers' => [
                    'Authorization' => 'Bearer ' . $this->apiKey,
                    'Content-Type' => 'application/json',
                ],
                'json' => [
                    'model' => $this->model,
                    'messages' => [
                        ['role' => 'user', 'content' => $prompt],
                    ],
                    'max_tokens' => $options['max_tokens'] ?? 4096,
                    'temperature' => $options['temperature'] ?? 0.7,
                ],
            ]);

            $data = json_decode($response->getBody()->getContents(), true);

            $latency = (microtime(true) - $startTime) * 1000;
            $inputTokens = $data['usage']['prompt_tokens'] ?? 0;
            $outputTokens = $data['usage']['completion_tokens'] ?? 0;
            $cost = $this->getEstimatedCost($inputTokens, $outputTokens);

            $this->recordSuccess($latency, $cost);
            $this->observability->endTrace('openai.invoke');

            return [
                'provider' => 'openai',
                'model' => $this->model,
                'content' => $data['choices'][0]['message']['content'] ?? '',
                'usage' => [
                    'input_tokens' => $inputTokens,
                    'output_tokens' => $outputTokens,
                ],
                'cost' => $cost,
                'latency' => $latency,
                'success' => true,
            ];

        } catch (GuzzleException $e) {
            $this->recordError();
            $this->observability->log('error', 'OpenAI provider error', [
                'error' => $e->getMessage(),
            ]);
            $this->observability->endTrace('openai.invoke');

            throw $e;
        }
    }

    public function streamInvoke(string $prompt, array $options = []): \Generator
    {
        try {
            $response = $this->client->post('/chat/completions', [
                'headers' => [
                    'Authorization' => 'Bearer ' . $this->apiKey,
                    'Content-Type' => 'application/json',
                ],
                'json' => [
                    'model' => $this->model,
                    'messages' => [
                        ['role' => 'user', 'content' => $prompt],
                    ],
                    'max_tokens' => $options['max_tokens'] ?? 4096,
                    'temperature' => $options['temperature'] ?? 0.7,
                    'stream' => true,
                ],
                'stream' => true,
            ]);

            $body = $response->getBody();

            while (!$body->eof()) {
                $line = $this->readLine($body);

                if (str_starts_with($line, 'data: ')) {
                    $json = substr($line, 6);

                    if ($json === '[DONE]') {
                        break;
                    }

                    $data = json_decode($json, true);

                    if (isset($data['choices'][0]['delta']['content'])) {
                        yield $data['choices'][0]['delta']['content'];
                    }
                }
            }

        } catch (GuzzleException $e) {
            $this->recordError();
            throw $e;
        }
    }

    private function readLine($stream): string
    {
        $line = '';

        while (!$stream->eof()) {
            $char = $stream->read(1);

            if ($char === "\n") {
                break;
            }

            $line .= $char;
        }

        return trim($line);
    }

    public function isAvailable(): bool
    {
        return $this->stats['error_rate'] < 0.5;
    }

    public function getEstimatedCost(int $inputTokens, int $outputTokens): float
    {
        $inputCostPer1k = 0.01;
        $outputCostPer1k = 0.03;

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

        $this->observability->recordMetric('openai.request.success', 1);
        $this->observability->recordMetric('openai.latency', $latency);
        $this->observability->recordMetric('openai.cost', $cost);
    }

    private function recordError(): void
    {
        $this->stats['total_requests'] = ($this->stats['total_requests'] ?? 0) + 1;
        $this->stats['error_rate'] = 1 - (($this->stats['successful_requests'] ?? 0) / $this->stats['total_requests']);

        $this->observability->recordMetric('openai.request.error', 1);
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
}
