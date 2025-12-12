<?php

declare(strict_types=1);

namespace MultiLLMGateway\Services;

use MultiLLMGateway\Providers\LLMProviderInterface;
use MultiLLMGateway\Routing\RouterInterface;
use MultiLLMGateway\MCP\MCPHandler;
use MultiLLMGateway\LLMOps\QualityEvaluator;

class GatewayService
{
    private array $providers = [];
    private RouterInterface $router;
    private MCPHandler $mcpHandler;
    private ?QualityEvaluator $qualityEvaluator;
    private ObservabilityService $observability;
    private bool $enableFallback;
    private bool $enableQualityMonitoring;

    public function __construct(
        RouterInterface $router,
        MCPHandler $mcpHandler,
        ObservabilityService $observability,
        ?QualityEvaluator $qualityEvaluator = null
    ) {
        $this->router = $router;
        $this->mcpHandler = $mcpHandler;
        $this->observability = $observability;
        $this->qualityEvaluator = $qualityEvaluator;
        $this->enableFallback = (bool) ($_ENV['ENABLE_FALLBACK'] ?? true);
        $this->enableQualityMonitoring = (bool) ($_ENV['ENABLE_QUALITY_MONITORING'] ?? true);
    }

    public function registerProvider(LLMProviderInterface $provider): void
    {
        $this->providers[$provider->getName()] = $provider;
    }

    public function invoke(string $prompt, array $options = []): array
    {
        $this->observability->startTrace('gateway.invoke');

        $useMCP = $options['use_mcp'] ?? true;
        $mcpContext = $options['mcp_context'] ?? [];

        if ($useMCP && !empty($mcpContext)) {
            $prompt = $this->mcpHandler->augmentPrompt($prompt, $mcpContext);
        }

        $provider = $this->router->selectProvider(
            array_values($this->providers),
            $prompt,
            $options
        );

        $attemptedProviders = [$provider->getName()];
        $errors = [];

        try {
            $response = $provider->invoke($prompt, $options);

            if ($useMCP) {
                $response = $this->mcpHandler->processResponse($response);
            }

            if ($this->enableQualityMonitoring && $this->qualityEvaluator) {
                $quality = $this->qualityEvaluator->evaluateResponse($prompt, $response['content']);
                $response['quality'] = $quality;

                if ($quality['overall_score'] < 5) {
                    $this->observability->log('warning', 'Low quality response detected', [
                        'provider' => $provider->getName(),
                        'score' => $quality['overall_score'],
                    ]);
                }
            }

            $this->observability->log('info', 'Gateway invocation successful', [
                'provider' => $provider->getName(),
                'latency' => $response['latency'],
                'cost' => $response['cost'],
            ]);

            $this->observability->recordMetric('gateway.request.success', 1);
            $this->observability->endTrace('gateway.invoke');

            return $response;

        } catch (\Exception $e) {
            $errors[$provider->getName()] = $e->getMessage();

            $this->observability->log('error', 'Primary provider failed', [
                'provider' => $provider->getName(),
                'error' => $e->getMessage(),
            ]);

            if ($this->enableFallback) {
                $fallbackResponse = $this->tryFallback(
                    $prompt,
                    $options,
                    $attemptedProviders,
                    $errors
                );

                if ($fallbackResponse) {
                    $this->observability->recordMetric('gateway.fallback.success', 1);
                    $this->observability->endTrace('gateway.invoke');
                    return $fallbackResponse;
                }
            }

            $this->observability->recordMetric('gateway.request.error', 1);
            $this->observability->endTrace('gateway.invoke');

            throw new \RuntimeException('All providers failed: ' . json_encode($errors));
        }
    }

    private function tryFallback(
        string $prompt,
        array $options,
        array &$attemptedProviders,
        array &$errors
    ): ?array {
        foreach ($this->providers as $provider) {
            if (in_array($provider->getName(), $attemptedProviders)) {
                continue;
            }

            if (!$provider->isAvailable()) {
                continue;
            }

            $attemptedProviders[] = $provider->getName();

            try {
                $this->observability->log('info', 'Attempting fallback provider', [
                    'provider' => $provider->getName(),
                ]);

                $response = $provider->invoke($prompt, $options);
                $response['fallback'] = true;
                $response['attempted_providers'] = $attemptedProviders;

                $this->observability->log('info', 'Fallback provider succeeded', [
                    'provider' => $provider->getName(),
                ]);

                return $response;

            } catch (\Exception $e) {
                $errors[$provider->getName()] = $e->getMessage();

                $this->observability->log('error', 'Fallback provider failed', [
                    'provider' => $provider->getName(),
                    'error' => $e->getMessage(),
                ]);
            }
        }

        return null;
    }

    public function streamInvoke(string $prompt, array $options = []): \Generator
    {
        $provider = $this->router->selectProvider(
            array_values($this->providers),
            $prompt,
            $options
        );

        try {
            yield from $provider->streamInvoke($prompt, $options);

        } catch (\Exception $e) {
            if ($this->enableFallback) {
                foreach ($this->providers as $fallbackProvider) {
                    if ($fallbackProvider->getName() === $provider->getName()) {
                        continue;
                    }

                    if (!$fallbackProvider->isAvailable()) {
                        continue;
                    }

                    try {
                        yield from $fallbackProvider->streamInvoke($prompt, $options);
                        return;
                    } catch (\Exception $fallbackError) {
                        continue;
                    }
                }
            }

            throw $e;
        }
    }

    public function getProviders(): array
    {
        return array_map(function ($provider) {
            return [
                'name' => $provider->getName(),
                'available' => $provider->isAvailable(),
                'avg_latency' => $provider->getAverageLatency(),
                'error_rate' => $provider->getErrorRate(),
            ];
        }, $this->providers);
    }

    public function healthCheck(): array
    {
        $results = [];

        foreach ($this->providers as $provider) {
            $results[$provider->getName()] = [
                'healthy' => $provider->healthCheck(),
                'available' => $provider->isAvailable(),
                'error_rate' => $provider->getErrorRate(),
            ];
        }

        return $results;
    }

    public function getMCPContext(): array
    {
        return $this->mcpHandler->getContext()->toArray();
    }
}
