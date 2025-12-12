<?php

declare(strict_types=1);

namespace MultiLLMGateway\MCP;

class MCPContext
{
    private string $version;
    private array $tools = [];
    private array $resources = [];
    private array $prompts = [];
    private array $metadata = [];

    public function __construct(string $version = '2024-11-05')
    {
        $this->version = $version;
    }

    public function addTool(string $name, string $description, array $parameters): self
    {
        $this->tools[] = [
            'name' => $name,
            'description' => $description,
            'inputSchema' => [
                'type' => 'object',
                'properties' => $parameters,
                'required' => array_keys(array_filter($parameters, fn($p) => $p['required'] ?? false)),
            ],
        ];

        return $this;
    }

    public function addResource(string $uri, string $name, string $mimeType, ?string $description = null): self
    {
        $this->resources[] = [
            'uri' => $uri,
            'name' => $name,
            'mimeType' => $mimeType,
            'description' => $description,
        ];

        return $this;
    }

    public function addPrompt(string $name, string $description, array $arguments = []): self
    {
        $this->prompts[] = [
            'name' => $name,
            'description' => $description,
            'arguments' => $arguments,
        ];

        return $this;
    }

    public function setMetadata(array $metadata): self
    {
        $this->metadata = $metadata;
        return $this;
    }

    public function toArray(): array
    {
        return [
            'protocol_version' => $this->version,
            'capabilities' => [
                'tools' => !empty($this->tools),
                'resources' => !empty($this->resources),
                'prompts' => !empty($this->prompts),
            ],
            'server_info' => array_merge([
                'name' => 'Multi-LLM Gateway',
                'version' => '1.0.0',
            ], $this->metadata),
            'tools' => $this->tools,
            'resources' => $this->resources,
            'prompts' => $this->prompts,
        ];
    }

    public function toJson(): string
    {
        return json_encode($this->toArray(), JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
    }
}
