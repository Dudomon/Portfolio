<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Middleware;

use LegalDocAnalyzer\Core\Request;
use LegalDocAnalyzer\Core\Response;
use LegalDocAnalyzer\Services\AuthService;

class AuthenticationMiddleware
{
    private AuthService $authService;

    public function __construct()
    {
        $this->authService = new AuthService();
    }

    public function handle(Request $request, callable $next): Response
    {
        $authHeader = $request->getHeader('Authorization');

        if (!$authHeader || !str_starts_with($authHeader, 'Bearer ')) {
            return Response::error('Missing or invalid authorization header', 401);
        }

        $token = substr($authHeader, 7);

        try {
            $payload = $this->authService->validateToken($token);
            $request->setAttribute('user', $payload);

            return $next($request);

        } catch (\Exception $e) {
            return Response::error('Invalid or expired token', 401);
        }
    }
}
