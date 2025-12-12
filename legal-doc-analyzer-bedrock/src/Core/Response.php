<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Core;

class Response
{
    private string $body;
    private int $statusCode;
    private array $headers;

    public function __construct(string $body, int $statusCode = 200, array $headers = [])
    {
        $this->body = $body;
        $this->statusCode = $statusCode;
        $this->headers = array_merge(['Content-Type' => 'application/json'], $headers);
    }

    public function getBody(): string
    {
        return $this->body;
    }

    public function getStatusCode(): int
    {
        return $this->statusCode;
    }

    public function getHeaders(): array
    {
        return $this->headers;
    }

    public static function json(array $data, int $statusCode = 200): self
    {
        return new self(
            json_encode($data, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES),
            $statusCode,
            ['Content-Type' => 'application/json']
        );
    }

    public static function success(array $data = [], string $message = 'Success'): self
    {
        return self::json([
            'success' => true,
            'message' => $message,
            'data' => $data,
            'timestamp' => date('c'),
        ], 200);
    }

    public static function error(string $message, int $statusCode = 400, array $errors = []): self
    {
        return self::json([
            'success' => false,
            'message' => $message,
            'errors' => $errors,
            'timestamp' => date('c'),
        ], $statusCode);
    }

    public static function created(array $data, string $message = 'Resource created'): self
    {
        return self::json([
            'success' => true,
            'message' => $message,
            'data' => $data,
            'timestamp' => date('c'),
        ], 201);
    }

    public static function noContent(): self
    {
        return new self('', 204);
    }
}
