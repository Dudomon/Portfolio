<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Core;

use LegalDocAnalyzer\Core\Router;
use LegalDocAnalyzer\Core\Request;
use LegalDocAnalyzer\Core\Response;

class Application
{
    private array $middlewares = [];
    private ?Router $router = null;

    public function addMiddleware(object $middleware): void
    {
        $this->middlewares[] = $middleware;
    }

    public function setRouter(Router $router): void
    {
        $this->router = $router;
    }

    public function run(): void
    {
        $request = Request::fromGlobals();

        $handler = function (Request $request) {
            return $this->router->dispatch($request);
        };

        foreach (array_reverse($this->middlewares) as $middleware) {
            $next = $handler;
            $handler = function (Request $request) use ($middleware, $next) {
                return $middleware->handle($request, $next);
            };
        }

        $response = $handler($request);
        $this->sendResponse($response);
    }

    private function sendResponse(Response $response): void
    {
        http_response_code($response->getStatusCode());

        foreach ($response->getHeaders() as $name => $value) {
            header("{$name}: {$value}");
        }

        echo $response->getBody();
    }
}
