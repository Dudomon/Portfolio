<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Middleware;

use LegalDocAnalyzer\Core\Request;
use LegalDocAnalyzer\Core\Response;
use LegalDocAnalyzer\Services\CacheService;
use LegalDocAnalyzer\Services\ObservabilityService;

class RateLimitMiddleware
{
    private CacheService $cache;
    private ObservabilityService $observability;
    private int $maxRequests;
    private int $windowSeconds;

    public function __construct()
    {
        $this->cache = new CacheService(new ObservabilityService());
        $this->observability = new ObservabilityService();
        $this->maxRequests = (int) $_ENV['RATE_LIMIT_REQUESTS'];
        $this->windowSeconds = (int) $_ENV['RATE_LIMIT_WINDOW'];
    }

    public function handle(Request $request, callable $next): Response
    {
        $identifier = $this->getIdentifier($request);
        $key = "rate_limit:{$identifier}";

        $current = (int) $this->cache->get($key) ?: 0;

        if ($current >= $this->maxRequests) {
            $this->observability->recordMetric('rate_limit.exceeded', 1);
            $this->observability->log('warning', 'Rate limit exceeded', [
                'identifier' => $identifier,
                'current' => $current,
                'limit' => $this->maxRequests,
            ]);

            return Response::error('Rate limit exceeded', 429, [
                'X-RateLimit-Limit' => (string) $this->maxRequests,
                'X-RateLimit-Remaining' => '0',
                'X-RateLimit-Reset' => (string) (time() + $this->windowSeconds),
            ]);
        }

        $this->cache->set($key, $current + 1, $this->windowSeconds);

        $response = $next($request);

        $remaining = $this->maxRequests - ($current + 1);

        $response = new Response(
            $response->getBody(),
            $response->getStatusCode(),
            array_merge($response->getHeaders(), [
                'X-RateLimit-Limit' => (string) $this->maxRequests,
                'X-RateLimit-Remaining' => (string) $remaining,
                'X-RateLimit-Reset' => (string) (time() + $this->windowSeconds),
            ])
        );

        return $response;
    }

    private function getIdentifier(Request $request): string
    {
        $authHeader = $request->getHeader('Authorization');

        if ($authHeader) {
            return hash('sha256', $authHeader);
        }

        $ip = $_SERVER['REMOTE_ADDR'] ?? 'unknown';
        $userAgent = $request->getHeader('User-Agent') ?? 'unknown';

        return hash('sha256', $ip . ':' . $userAgent);
    }
}
