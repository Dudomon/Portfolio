<?php

declare(strict_types=1);

namespace MultiLLMGateway\Routing;

use MultiLLMGateway\Providers\LLMProviderInterface;

interface RouterInterface
{
    public function selectProvider(array $providers, string $prompt, array $options = []): LLMProviderInterface;

    public function getName(): string;
}
