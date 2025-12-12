<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Middleware;

use LegalDocAnalyzer\Core\Request;
use LegalDocAnalyzer\Core\Response;
use LegalDocAnalyzer\Services\ObservabilityService;

class ObservabilityMiddleware
{
    private ObservabilityService $observability;

    public function __construct()
    {
        $this->observability = new ObservabilityService();
    }

    public function handle(Request $request, callable $next): Response
    {
        $startTime = microtime(true);
        $requestId = uniqid('req_');

        $_SERVER['HTTP_X_REQUEST_ID'] = $requestId;

        $this->observability->startTrace('http.request');
        $this->observability->log('info', 'Request received', [
            'request_id' => $requestId,
            'method' => $request->getMethod(),
            'path' => $request->getPath(),
            'ip' => $_SERVER['REMOTE_ADDR'] ?? 'unknown',
        ]);

        $response = $next($request);

        $duration = (microtime(true) - $startTime) * 1000;

        $this->observability->recordApiCall(
            $request->getPath(),
            $request->getMethod(),
            $response->getStatusCode(),
            $duration
        );

        $this->observability->log('info', 'Request completed', [
            'request_id' => $requestId,
            'status_code' => $response->getStatusCode(),
            'duration_ms' => $duration,
        ]);

        $this->observability->endTrace('http.request');

        return new Response(
            $response->getBody(),
            $response->getStatusCode(),
            array_merge($response->getHeaders(), [
                'X-Request-ID' => $requestId,
                'X-Response-Time' => (string) round($duration, 2) . 'ms',
            ])
        );
    }
}
