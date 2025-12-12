<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Services;

use Predis\Client as RedisClient;
use LegalDocAnalyzer\Services\ObservabilityService;

class CacheService
{
    private RedisClient $redis;
    private ObservabilityService $observability;
    private int $defaultTtl;

    public function __construct(ObservabilityService $observability)
    {
        $this->redis = new RedisClient([
            'scheme' => 'tcp',
            'host' => $_ENV['REDIS_HOST'],
            'port' => (int) $_ENV['REDIS_PORT'],
            'password' => $_ENV['REDIS_PASSWORD'] ?: null,
            'database' => (int) $_ENV['REDIS_DB'],
        ]);

        $this->observability = $observability;
        $this->defaultTtl = (int) $_ENV['CACHE_TTL'];
    }

    public function get(string $key): mixed
    {
        try {
            $value = $this->redis->get($key);

            if ($value === null) {
                return null;
            }

            return json_decode($value, true);

        } catch (\Exception $e) {
            $this->observability->log('error', 'Cache get error', [
                'key' => $key,
                'error' => $e->getMessage(),
            ]);

            return null;
        }
    }

    public function set(string $key, mixed $value, ?int $ttl = null): bool
    {
        try {
            $ttl = $ttl ?? $this->defaultTtl;
            $encoded = json_encode($value);

            $this->redis->setex($key, $ttl, $encoded);

            return true;

        } catch (\Exception $e) {
            $this->observability->log('error', 'Cache set error', [
                'key' => $key,
                'error' => $e->getMessage(),
            ]);

            return false;
        }
    }

    public function delete(string $key): bool
    {
        try {
            $this->redis->del([$key]);
            return true;

        } catch (\Exception $e) {
            $this->observability->log('error', 'Cache delete error', [
                'key' => $key,
                'error' => $e->getMessage(),
            ]);

            return false;
        }
    }

    public function has(string $key): bool
    {
        try {
            return $this->redis->exists($key) > 0;
        } catch (\Exception $e) {
            return false;
        }
    }

    public function invalidatePattern(string $pattern): int
    {
        try {
            $keys = $this->redis->keys($pattern);

            if (empty($keys)) {
                return 0;
            }

            return $this->redis->del($keys);

        } catch (\Exception $e) {
            $this->observability->log('error', 'Cache invalidate pattern error', [
                'pattern' => $pattern,
                'error' => $e->getMessage(),
            ]);

            return 0;
        }
    }
}
