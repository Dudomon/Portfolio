<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Core;

use LegalDocAnalyzer\Core\Request;
use LegalDocAnalyzer\Core\Response;

class Router
{
    private array $routes = [];
    private array $groupMiddlewares = [];
    private string $groupPrefix = '';

    public function get(string $path, array $handler): void
    {
        $this->addRoute('GET', $path, $handler);
    }

    public function post(string $path, array $handler): void
    {
        $this->addRoute('POST', $path, $handler);
    }

    public function put(string $path, array $handler): void
    {
        $this->addRoute('PUT', $path, $handler);
    }

    public function delete(string $path, array $handler): void
    {
        $this->addRoute('DELETE', $path, $handler);
    }

    public function group(string $prefix, array $middlewares, callable $callback): void
    {
        $previousPrefix = $this->groupPrefix;
        $previousMiddlewares = $this->groupMiddlewares;

        $this->groupPrefix = $previousPrefix . $prefix;
        $this->groupMiddlewares = array_merge($previousMiddlewares, $middlewares);

        $callback($this);

        $this->groupPrefix = $previousPrefix;
        $this->groupMiddlewares = $previousMiddlewares;
    }

    private function addRoute(string $method, string $path, array $handler): void
    {
        $fullPath = $this->groupPrefix . $path;
        $this->routes[] = [
            'method' => $method,
            'path' => $fullPath,
            'pattern' => $this->pathToPattern($fullPath),
            'handler' => $handler,
            'middlewares' => $this->groupMiddlewares,
        ];
    }

    private function pathToPattern(string $path): string
    {
        return '#^' . preg_replace('/\{([a-zA-Z0-9_]+)\}/', '(?P<$1>[^/]+)', $path) . '$#';
    }

    public function dispatch(Request $request): Response
    {
        $method = $request->getMethod();
        $path = $request->getPath();

        foreach ($this->routes as $route) {
            if ($route['method'] !== $method) {
                continue;
            }

            if (preg_match($route['pattern'], $path, $matches)) {
                $params = array_filter($matches, 'is_string', ARRAY_FILTER_USE_KEY);
                $request->setParams($params);

                $handler = function (Request $request) use ($route) {
                    [$class, $method] = $route['handler'];
                    $controller = new $class();
                    return $controller->$method($request);
                };

                foreach (array_reverse($route['middlewares']) as $middleware) {
                    $next = $handler;
                    $handler = function (Request $request) use ($middleware, $next) {
                        return $middleware->handle($request, $next);
                    };
                }

                return $handler($request);
            }
        }

        return new Response(
            json_encode(['error' => 'Not Found']),
            404,
            ['Content-Type' => 'application/json']
        );
    }
}
