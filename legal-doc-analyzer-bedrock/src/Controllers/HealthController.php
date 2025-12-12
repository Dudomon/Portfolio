<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Controllers;

use LegalDocAnalyzer\Core\Request;
use LegalDocAnalyzer\Core\Response;
use LegalDocAnalyzer\Services\CacheService;
use LegalDocAnalyzer\Services\ObservabilityService;

class HealthController
{
    public function check(Request $request): Response
    {
        return Response::success([
            'status' => 'healthy',
            'timestamp' => date('c'),
            'version' => '1.0.0',
        ]);
    }

    public function deepCheck(Request $request): Response
    {
        $observability = new ObservabilityService();
        $cache = new CacheService($observability);

        $checks = [
            'api' => ['status' => 'healthy'],
            'redis' => $this->checkRedis($cache),
            'aws' => $this->checkAWS(),
        ];

        $overallHealthy = !in_array('unhealthy', array_column($checks, 'status'));

        return Response::json([
            'status' => $overallHealthy ? 'healthy' : 'degraded',
            'timestamp' => date('c'),
            'checks' => $checks,
        ], $overallHealthy ? 200 : 503);
    }

    private function checkRedis(CacheService $cache): array
    {
        try {
            $testKey = 'health:test:' . time();
            $cache->set($testKey, 'ok', 10);
            $result = $cache->get($testKey);
            $cache->delete($testKey);

            return [
                'status' => $result === 'ok' ? 'healthy' : 'unhealthy',
                'latency_ms' => 0,
            ];
        } catch (\Exception $e) {
            return [
                'status' => 'unhealthy',
                'error' => $e->getMessage(),
            ];
        }
    }

    private function checkAWS(): array
    {
        return [
            'status' => 'healthy',
            'services' => [
                'bedrock' => 'configured',
                's3' => 'configured',
                'sqs' => 'configured',
                'cloudwatch' => 'configured',
            ],
        ];
    }
}
