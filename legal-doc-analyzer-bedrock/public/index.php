<?php

declare(strict_types=1);

require_once __DIR__ . '/../vendor/autoload.php';

use LegalDocAnalyzer\Core\Application;
use LegalDocAnalyzer\Core\Router;
use LegalDocAnalyzer\Middleware\AuthenticationMiddleware;
use LegalDocAnalyzer\Middleware\RateLimitMiddleware;
use LegalDocAnalyzer\Middleware\ObservabilityMiddleware;
use LegalDocAnalyzer\Middleware\ErrorHandlerMiddleware;
use LegalDocAnalyzer\Controllers\DocumentController;
use LegalDocAnalyzer\Controllers\AnalysisController;
use LegalDocAnalyzer\Controllers\AuthController;
use LegalDocAnalyzer\Controllers\HealthController;

Dotenv\Dotenv::createImmutable(__DIR__ . '/../')->load();

$app = new Application();

$app->addMiddleware(new ErrorHandlerMiddleware());
$app->addMiddleware(new ObservabilityMiddleware());
$app->addMiddleware(new RateLimitMiddleware());

$router = new Router();

$router->post('/api/v1/auth/login', [AuthController::class, 'login']);
$router->post('/api/v1/auth/refresh', [AuthController::class, 'refresh']);

$router->group('/api/v1', [new AuthenticationMiddleware()], function (Router $router) {
    $router->post('/documents/upload', [DocumentController::class, 'upload']);
    $router->get('/documents/{id}', [DocumentController::class, 'get']);
    $router->get('/documents', [DocumentController::class, 'list']);
    $router->delete('/documents/{id}', [DocumentController::class, 'delete']);

    $router->post('/analysis/extract', [AnalysisController::class, 'extractEntities']);
    $router->post('/analysis/classify', [AnalysisController::class, 'classify']);
    $router->post('/analysis/summarize', [AnalysisController::class, 'summarize']);
    $router->post('/analysis/compare', [AnalysisController::class, 'compare']);
    $router->get('/analysis/{id}', [AnalysisController::class, 'getResult']);
    $router->get('/analysis/{id}/status', [AnalysisController::class, 'getStatus']);
});

$router->get('/health', [HealthController::class, 'check']);
$router->get('/health/deep', [HealthController::class, 'deepCheck']);

$app->setRouter($router);
$app->run();
