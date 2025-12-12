<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Controllers;

use LegalDocAnalyzer\Core\Request;
use LegalDocAnalyzer\Core\Response;
use LegalDocAnalyzer\Services\AuthService;

class AuthController
{
    private AuthService $authService;

    public function __construct()
    {
        $this->authService = new AuthService();
    }

    public function login(Request $request): Response
    {
        $username = $request->getBodyParam('username');
        $password = $request->getBodyParam('password');

        if (!$username || !$password) {
            return Response::error('Username and password are required', 400);
        }

        $user = $this->authService->authenticate($username, $password);

        if (!$user) {
            return Response::error('Invalid credentials', 401);
        }

        $token = $this->authService->generateToken($user);

        return Response::success([
            'token' => $token,
            'user' => [
                'id' => $user['id'],
                'username' => $user['username'],
                'email' => $user['email'],
                'roles' => $user['roles'],
            ],
        ], 'Login successful');
    }

    public function refresh(Request $request): Response
    {
        $authHeader = $request->getHeader('Authorization');

        if (!$authHeader || !str_starts_with($authHeader, 'Bearer ')) {
            return Response::error('Missing or invalid authorization header', 401);
        }

        $token = substr($authHeader, 7);

        try {
            $newToken = $this->authService->refreshToken($token);

            return Response::success([
                'token' => $newToken,
            ], 'Token refreshed successfully');

        } catch (\Exception $e) {
            return Response::error('Invalid or expired token', 401);
        }
    }
}
