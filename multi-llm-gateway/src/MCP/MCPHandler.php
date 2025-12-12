<?php

declare(strict_types=1);

namespace MultiLLMGateway\MCP;

use MultiLLMGateway\Services\ObservabilityService;

class MCPHandler
{
    private ObservabilityService $observability;
    private MCPContext $context;
    private array $toolHandlers = [];

    public function __construct(ObservabilityService $observability)
    {
        $this->observability = $observability;
        $this->context = new MCPContext();
        $this->registerDefaultTools();
    }

    private function registerDefaultTools(): void
    {
        $this->context->addTool(
            'analyze_sentiment',
            'Analyze sentiment of given text',
            [
                'text' => [
                    'type' => 'string',
                    'description' => 'Text to analyze',
                    'required' => true,
                ],
            ]
        );

        $this->context->addTool(
            'extract_entities',
            'Extract named entities from text',
            [
                'text' => [
                    'type' => 'string',
                    'description' => 'Text to extract entities from',
                    'required' => true,
                ],
                'types' => [
                    'type' => 'array',
                    'description' => 'Entity types to extract (person, org, location)',
                    'required' => false,
                ],
            ]
        );

        $this->context->addTool(
            'summarize',
            'Generate a summary of text',
            [
                'text' => [
                    'type' => 'string',
                    'description' => 'Text to summarize',
                    'required' => true,
                ],
                'max_length' => [
                    'type' => 'integer',
                    'description' => 'Maximum summary length',
                    'required' => false,
                ],
            ]
        );

        $this->context->addPrompt(
            'legal_analysis',
            'Analyze legal document with structured output',
            [
                [
                    'name' => 'document_type',
                    'description' => 'Type of legal document',
                    'required' => true,
                ],
            ]
        );

        $this->context->addPrompt(
            'code_review',
            'Review code with best practices check',
            [
                [
                    'name' => 'language',
                    'description' => 'Programming language',
                    'required' => true,
                ],
            ]
        );
    }

    public function getContext(): MCPContext
    {
        return $this->context;
    }

    public function registerTool(string $name, callable $handler): void
    {
        $this->toolHandlers[$name] = $handler;
    }

    public function executeTool(string $name, array $arguments): array
    {
        if (!isset($this->toolHandlers[$name])) {
            throw new \RuntimeException("Tool not found: {$name}");
        }

        $this->observability->log('info', 'Executing MCP tool', [
            'tool' => $name,
            'arguments' => $arguments,
        ]);

        $this->observability->startTrace("mcp.tool.{$name}");

        try {
            $result = $this->toolHandlers[$name]($arguments);

            $this->observability->recordMetric('mcp.tool.success', 1);
            $this->observability->endTrace("mcp.tool.{$name}");

            return [
                'success' => true,
                'result' => $result,
            ];

        } catch (\Exception $e) {
            $this->observability->recordMetric('mcp.tool.error', 1);
            $this->observability->log('error', 'MCP tool execution failed', [
                'tool' => $name,
                'error' => $e->getMessage(),
            ]);
            $this->observability->endTrace("mcp.tool.{$name}");

            return [
                'success' => false,
                'error' => $e->getMessage(),
            ];
        }
    }

    public function listTools(): array
    {
        return array_keys($this->toolHandlers);
    }

    public function augmentPrompt(string $basePrompt, array $mcpContext): string
    {
        if (empty($mcpContext)) {
            return $basePrompt;
        }

        $contextSection = "\n\n=== Available Tools ===\n";

        foreach ($this->context->toArray()['tools'] as $tool) {
            $contextSection .= "\n- {$tool['name']}: {$tool['description']}";
        }

        $contextSection .= "\n\n=== Context ===\n";
        $contextSection .= json_encode($mcpContext, JSON_PRETTY_PRINT);

        return $basePrompt . $contextSection;
    }

    public function processResponse(array $response): array
    {
        if (!isset($response['content'])) {
            return $response;
        }

        if (preg_match('/<tool_use>(.*?)<\/tool_use>/s', $response['content'], $matches)) {
            $toolData = json_decode($matches[1], true);

            if ($toolData && isset($toolData['name'])) {
                $toolResult = $this->executeTool($toolData['name'], $toolData['arguments'] ?? []);

                $response['tool_call'] = [
                    'name' => $toolData['name'],
                    'arguments' => $toolData['arguments'] ?? [],
                    'result' => $toolResult,
                ];

                $response['content'] = str_replace($matches[0], json_encode($toolResult), $response['content']);
            }
        }

        return $response;
    }
}
