<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Services;

use Aws\CloudWatchLogs\CloudWatchLogsClient;
use Aws\CloudWatch\CloudWatchClient;
use Aws\XRay\XRayClient;
use Monolog\Logger;
use Monolog\Handler\StreamHandler;
use Monolog\Formatter\JsonFormatter;

class ObservabilityService
{
    private Logger $logger;
    private CloudWatchLogsClient $cloudWatchLogs;
    private CloudWatchClient $cloudWatch;
    private ?XRayClient $xray;
    private bool $enabled;
    private string $logGroup;
    private string $logStream;
    private string $metricsNamespace;
    private array $traces = [];

    public function __construct()
    {
        $this->enabled = (bool) ($_ENV['OBSERVABILITY_ENABLED'] ?? true);
        $this->logGroup = $_ENV['CLOUDWATCH_LOG_GROUP'];
        $this->logStream = $_ENV['CLOUDWATCH_LOG_STREAM'] . '-' . date('Y-m-d-H');
        $this->metricsNamespace = $_ENV['METRICS_NAMESPACE'];

        $this->logger = new Logger('legal-doc-analyzer');
        $handler = new StreamHandler('php://stdout', Logger::DEBUG);
        $handler->setFormatter(new JsonFormatter());
        $this->logger->pushHandler($handler);

        if ($this->enabled) {
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

            if ($_ENV['XRAY_ENABLED'] ?? false) {
                $this->xray = new XRayClient($awsConfig);
            }
        }
    }

    public function log(string $level, string $message, array $context = []): void
    {
        $context = array_merge($context, [
            'timestamp' => date('c'),
            'environment' => $_ENV['APP_ENV'],
            'request_id' => $_SERVER['HTTP_X_REQUEST_ID'] ?? uniqid('req_'),
        ]);

        $this->logger->log($level, $message, $context);

        if ($this->enabled) {
            $this->sendToCloudWatch($level, $message, $context);
        }
    }

    private function sendToCloudWatch(string $level, string $message, array $context): void
    {
        try {
            $this->cloudWatchLogs->putLogEvents([
                'logGroupName' => $this->logGroup,
                'logStreamName' => $this->logStream,
                'logEvents' => [
                    [
                        'message' => json_encode([
                            'level' => $level,
                            'message' => $message,
                            'context' => $context,
                        ]),
                        'timestamp' => (int) (microtime(true) * 1000),
                    ],
                ],
            ]);
        } catch (\Exception $e) {
            $this->logger->error('Failed to send logs to CloudWatch', [
                'error' => $e->getMessage(),
            ]);
        }
    }

    public function recordMetric(string $name, float $value, string $unit = 'None'): void
    {
        if (!$this->enabled) {
            return;
        }

        try {
            $this->cloudWatch->putMetricData([
                'Namespace' => $this->metricsNamespace,
                'MetricData' => [
                    [
                        'MetricName' => $name,
                        'Value' => $value,
                        'Unit' => $unit,
                        'Timestamp' => time(),
                        'Dimensions' => [
                            [
                                'Name' => 'Environment',
                                'Value' => $_ENV['APP_ENV'],
                            ],
                        ],
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
        $this->traces[$name] = [
            'start_time' => microtime(true),
            'name' => $name,
        ];
    }

    public function endTrace(string $name): void
    {
        if (!isset($this->traces[$name])) {
            return;
        }

        $trace = $this->traces[$name];
        $duration = (microtime(true) - $trace['start_time']) * 1000;

        $this->recordMetric("{$name}.duration", $duration, 'Milliseconds');

        if ($this->xray) {
            $this->sendTraceToXRay($name, $trace['start_time'], $duration);
        }

        unset($this->traces[$name]);
    }

    private function sendTraceToXRay(string $name, float $startTime, float $duration): void
    {
        try {
            $segment = [
                'name' => $name,
                'id' => bin2hex(random_bytes(8)),
                'trace_id' => $this->getTraceId(),
                'start_time' => $startTime,
                'end_time' => $startTime + ($duration / 1000),
            ];

            $this->xray->putTraceSegments([
                'TraceSegmentDocuments' => [json_encode($segment)],
            ]);
        } catch (\Exception $e) {
            $this->logger->error('Failed to send trace to X-Ray', [
                'trace' => $name,
                'error' => $e->getMessage(),
            ]);
        }
    }

    private function getTraceId(): string
    {
        if (isset($_SERVER['HTTP_X_AMZN_TRACE_ID'])) {
            return $_SERVER['HTTP_X_AMZN_TRACE_ID'];
        }

        return '1-' . dechex(time()) . '-' . bin2hex(random_bytes(12));
    }

    public function recordApiCall(string $endpoint, string $method, int $statusCode, float $duration): void
    {
        $this->recordMetric('api.request.count', 1, 'Count');
        $this->recordMetric('api.request.duration', $duration, 'Milliseconds');
        $this->recordMetric("api.response.{$statusCode}", 1, 'Count');

        $this->log('info', 'API request completed', [
            'endpoint' => $endpoint,
            'method' => $method,
            'status_code' => $statusCode,
            'duration_ms' => $duration,
        ]);
    }
}
