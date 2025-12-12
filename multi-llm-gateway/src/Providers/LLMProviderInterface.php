<?php

declare(strict_types=1);

namespace MultiLLMGateway\Providers;

interface LLMProviderInterface
{
    public function getName(): string;

    public function invoke(string $prompt, array $options = []): array;

    public function streamInvoke(string $prompt, array $options = []): \Generator;

    public function isAvailable(): bool;

    public function getEstimatedCost(int $inputTokens, int $outputTokens): float;

    public function getAverageLatency(): float;

    public function getErrorRate(): float;

    public function healthCheck(): bool;
}
