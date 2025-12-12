<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Middleware;

use LegalDocAnalyzer\Core\Request;
use LegalDocAnalyzer\Core\Response;
use LegalDocAnalyzer\Services\ObservabilityService;

class ErrorHandlerMiddleware
{
    private ObservabilityService $observability;

    public function __construct()
    {
        $this->observability = new ObservabilityService();
    }

    public function handle(Request $request, callable $next): Response
    {
        try {
            return $next($request);

        } catch (\Throwable $e) {
            $this->observability->recordMetric('error.caught', 1);
            $this->observability->log('error', 'Unhandled exception', [
                'exception' => get_class($e),
                'message' => $e->getMessage(),
                'file' => $e->getFile(),
                'line' => $e->getLine(),
                'trace' => $e->getTraceAsString(),
            ]);

            $statusCode = method_exists($e, 'getStatusCode') ? $e->getStatusCode() : 500;
            $message = $_ENV['APP_DEBUG'] === 'true' ? $e->getMessage() : 'Internal server error';

            $errorData = [
                'error' => get_class($e),
                'message' => $message,
            ];

            if ($_ENV['APP_DEBUG'] === 'true') {
                $errorData['file'] = $e->getFile();
                $errorData['line'] = $e->getLine();
                $errorData['trace'] = explode("\n", $e->getTraceAsString());
            }

            return Response::error($message, $statusCode, $errorData);
        }
    }
}
