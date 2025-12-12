<?php

declare(strict_types=1);

namespace MultiLLMGateway\Services;

use Aws\CloudWatchLogs\CloudWatchLogsClient;
use Aws\CloudWatch\CloudWatchClient;
use Monolog\Logger;
use Monolog\Handler\StreamHandler;
use Monolog\Formatter\JsonFormatter;

class ObservabilityService
{
    private Logger $logger;
    private ?CloudWatchLogsClient $cloudWatchLogs;
    private ?CloudWatchClient $cloudWatch;
    private bool $enabled;
    private array $traces = [];

    public function __construct()
    {
        $this->enabled = (bool) ($_ENV['OBSERVABILITY_ENABLED'] ?? true);

        $this->logger = new Logger('multi-llm-gateway');
        $handler = new StreamHandler('php://stdout', Logger::DEBUG);
        $handler->setFormatter(new JsonFormatter());
        $this->logger->pushHandler($handler);

        if ($this->enabled && isset($_ENV['AWS_ACCESS_KEY_ID'])) {
            $awsConfig = [
                'region' => $_ENV['AWS_REGION'],
                'version' => 'latest',
                'credentials' => [
                    'key' => $_ENV['AWS_ACCESS_KEY_ID'],
                    'secret' => $_ENV['AWS_SECRET_ACCESS_KEY'],
                ],
            ];

            $this->cloudWatchLogs = new CloudWatchLogsClient($awsConfig);
            $this->cloudWatch = new CloudWatchClient($awsConfig);
        }
    }

    public function log(string $level, string $message, array $context = []): void
    {
        $context = array_merge($context, [
            'timestamp' => date('c'),
            'environment' => $_ENV['APP_ENV'] ?? 'production',
        ]);

        $this->logger->log($level, $message, $context);
    }

    public function recordMetric(string $name, float $value, string $unit = 'None'): void
    {
        if (!$this->enabled || !$this->cloudWatch) {
            return;
        }

        try {
            $this->cloudWatch->putMetricData([
                'Namespace' => $_ENV['METRICS_NAMESPACE'] ?? 'MultiLLMGateway',
                'MetricData' => [
                    [
                        'MetricName' => $name,
                        'Value' => $value,
                        'Unit' => $unit,
                        'Timestamp' => time(),
                    ],
                ],
            ]);
        } catch (\Exception $e) {
            $this->logger->error('Failed to record metric', [
                'metric' => $name,
                'error' => $e->getMessage(),
            ]);
        }
    }

    public function startTrace(string $name): void
    {
        $this->traces[$name] = microtime(true);
    }

    public function endTrace(string $name): void
    {
        if (!isset($this->traces[$name])) {
            return;
        }

        $duration = (microtime(true) - $this->traces[$name]) * 1000;
        $this->recordMetric("{$name}.duration", $duration, 'Milliseconds');

        unset($this->traces[$name]);
    }
}
