<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Core;

class Request
{
    private string $method;
    private string $path;
    private array $headers;
    private array $query;
    private array $body;
    private array $params = [];
    private array $attributes = [];

    public function __construct(
        string $method,
        string $path,
        array $headers = [],
        array $query = [],
        array $body = []
    ) {
        $this->method = $method;
        $this->path = $path;
        $this->headers = $headers;
        $this->query = $query;
        $this->body = $body;
    }

    public static function fromGlobals(): self
    {
        $headers = getallheaders() ?: [];
        $method = $_SERVER['REQUEST_METHOD'] ?? 'GET';
        $path = parse_url($_SERVER['REQUEST_URI'] ?? '/', PHP_URL_PATH);
        $query = $_GET;

        $body = [];
        $contentType = $headers['Content-Type'] ?? '';

        if (str_contains($contentType, 'application/json')) {
            $body = json_decode(file_get_contents('php://input'), true) ?? [];
        } else {
            $body = $_POST;
        }

        return new self($method, $path, $headers, $query, $body);
    }

    public function getMethod(): string
    {
        return $this->method;
    }

    public function getPath(): string
    {
        return $this->path;
    }

    public function getHeaders(): array
    {
        return $this->headers;
    }

    public function getHeader(string $name): ?string
    {
        return $this->headers[$name] ?? null;
    }

    public function getQuery(): array
    {
        return $this->query;
    }

    public function getQueryParam(string $name, mixed $default = null): mixed
    {
        return $this->query[$name] ?? $default;
    }

    public function getBody(): array
    {
        return $this->body;
    }

    public function getBodyParam(string $name, mixed $default = null): mixed
    {
        return $this->body[$name] ?? $default;
    }

    public function setParams(array $params): void
    {
        $this->params = $params;
    }

    public function getParams(): array
    {
        return $this->params;
    }

    public function getParam(string $name, mixed $default = null): mixed
    {
        return $this->params[$name] ?? $default;
    }

    public function setAttribute(string $name, mixed $value): void
    {
        $this->attributes[$name] = $value;
    }

    public function getAttribute(string $name, mixed $default = null): mixed
    {
        return $this->attributes[$name] ?? $default;
    }

    public function getFile(string $name): ?array
    {
        return $_FILES[$name] ?? null;
    }
}
