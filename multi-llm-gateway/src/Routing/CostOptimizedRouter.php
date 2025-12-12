<?php

declare(strict_types=1);

namespace MultiLLMGateway\Routing;

use MultiLLMGateway\Providers\LLMProviderInterface;
use MultiLLMGateway\Services\ObservabilityService;

class CostOptimizedRouter implements RouterInterface
{
    private ObservabilityService $observability;

    public function __construct(ObservabilityService $observability)
    {
        $this->observability = $observability;
    }

    public function getName(): string
    {
        return 'cost_optimized';
    }

    public function selectProvider(array $providers, string $prompt, array $options = []): LLMProviderInterface
    {
        $estimatedTokens = $this->estimateTokens($prompt);
        $scores = [];

        foreach ($providers as $provider) {
            if (!$provider->isAvailable()) {
                $this->observability->log('info', 'Provider not available', [
                    'provider' => $provider->getName(),
                ]);
                continue;
            }

            $estimatedCost = $provider->getEstimatedCost($estimatedTokens['input'], $estimatedTokens['output']);
            $latency = $provider->getAverageLatency();
            $errorRate = $provider->getErrorRate();

            $costScore = 1 - min($estimatedCost / 0.05, 1);
            $latencyScore = 1 - min($latency / 5000, 1);
            $reliabilityScore = 1 - $errorRate;

            $totalScore = ($costScore * 0.6) + ($latencyScore * 0.2) + ($reliabilityScore * 0.2);

            $scores[$provider->getName()] = [
                'provider' => $provider,
                'score' => $totalScore,
                'estimated_cost' => $estimatedCost,
                'latency' => $latency,
                'error_rate' => $errorRate,
            ];
        }

        if (empty($scores)) {
            throw new \RuntimeException('No available providers');
        }

        usort($scores, function ($a, $b) {
            return $b['score'] <=> $a['score'];
        });

        $selected = $scores[0];

        $this->observability->log('info', 'Provider selected by cost optimizer', [
            'provider' => $selected['provider']->getName(),
            'score' => $selected['score'],
            'estimated_cost' => $selected['estimated_cost'],
            'latency' => $selected['latency'],
            'error_rate' => $selected['error_rate'],
        ]);

        $this->observability->recordMetric('router.selection', 1);
        $this->observability->recordMetric('router.estimated_cost', $selected['estimated_cost']);

        return $selected['provider'];
    }

    private function estimateTokens(string $prompt): array
    {
        $words = str_word_count($prompt);
        $inputTokens = (int) ceil($words * 1.3);
        $outputTokens = min((int) ceil($inputTokens * 0.3), 1000);

        return [
            'input' => $inputTokens,
            'output' => $outputTokens,
        ];
    }
}
